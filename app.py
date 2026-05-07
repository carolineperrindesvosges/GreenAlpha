import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

# ── Config page ──────────────────────────────────────────────────────────
st.set_page_config(page_title="GreenAlpha Challenge", layout="wide")

C_BLUE = '#2C3E7A'
C_GREEN = '#27AE60'
C_RED = '#E74C3C'
C_GOLD = '#F39C12'
C_BG = '#FAFAFA'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': C_BG,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Univers d'actifs ─────────────────────────────────────────────────────
ASSETS = {
    'TotalEnergies': {'ret': 0.082, 'vol': 0.221, 'esg': 32, 'sector': 'Énergie'},
    'Schneider':     {'ret': 0.134, 'vol': 0.198, 'esg': 78, 'sector': 'Industrie'},
    'Air Liquide':   {'ret': 0.096, 'vol': 0.172, 'esg': 65, 'sector': 'Chimie'},
    'BNP Paribas':   {'ret': 0.071, 'vol': 0.263, 'esg': 44, 'sector': 'Finance'},
    'Danone':        {'ret': 0.058, 'vol': 0.189, 'esg': 71, 'sector': 'Conso.'},
    'Stellantis':    {'ret': 0.105, 'vol': 0.298, 'esg': 29, 'sector': 'Auto'},
    "L'Oréal":      {'ret': 0.118, 'vol': 0.181, 'esg': 82, 'sector': 'Luxe'},
    'Vinci':         {'ret': 0.089, 'vol': 0.207, 'esg': 56, 'sector': 'Infra'},
    'Sanofi':        {'ret': 0.076, 'vol': 0.194, 'esg': 68, 'sector': 'Santé'},
    'Engie':         {'ret': 0.063, 'vol': 0.241, 'esg': 51, 'sector': 'Utilities'},
}

NAMES = list(ASSETS.keys())
N = len(NAMES)
RETS = np.array([ASSETS[a]['ret'] for a in NAMES])
VOLS = np.array([ASSETS[a]['vol'] for a in NAMES])
ESG = np.array([ASSETS[a]['esg'] for a in NAMES])
RF = 0.025
ESG_MIN = 55

# ── Matrice de covariance stable ─────────────────────────────────────────
np.random.seed(42)
_A = np.random.randn(N, N) * 0.15
_C = np.eye(N) + _A + _A.T
_C /= np.abs(_C).max(axis=1, keepdims=True)
np.fill_diagonal(_C, 1.0)
_C = (_C + _C.T) / 2
_ev = np.linalg.eigvals(_C)
if _ev.min() < 0:
    _C += (-_ev.min() + 0.01) * np.eye(N)
_C /= np.sqrt(np.outer(np.diag(_C), np.diag(_C)))
np.fill_diagonal(_C, 1.0)
COV = np.outer(VOLS, VOLS) * _C


# ── Fonctions ────────────────────────────────────────────────────────────
def get_asset_df(rets=None, esg=None):
    if rets is None:
        rets = RETS
    if esg is None:
        esg = ESG
    return pd.DataFrame({
        'Actif': NAMES,
        'Secteur': [ASSETS[a]['sector'] for a in NAMES],
        'Rendement (%)': (rets * 100).round(1),
        'Volatilité (%)': (VOLS * 100).round(1),
        'Score ESG /100': esg,
    })


def style_assets(df: pd.DataFrame):
    def color_esg(val):
        if val >= 65:
            return 'background-color:#d4edda;color:#155724;font-weight:bold'
        if val >= 45:
            return 'background-color:#fff3cd;color:#856404'
        return 'background-color:#f8d7da;color:#721c24'

    return df.style.applymap(color_esg, subset=['Score ESG /100'])


def port_stats(w, rets=None):
    if rets is None:
        rets = RETS
    return float(w @ rets), float(np.sqrt(w @ COV @ w))


def optimize(esg_min=None, rf=RF, rets=None, esg_scores=None):
    if rets is None:
        rets = RETS
    if esg_scores is None:
        esg_scores = ESG

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if esg_min is not None:
        cons.append({'type': 'ineq', 'fun': lambda w, s=esg_min: w @ esg_scores - s})

    def neg_sharpe(w):
        r, v = port_stats(w, rets)
        return -(r - rf) / v if v > 1e-8 else 0

    res = minimize(
        neg_sharpe,
        x0=np.ones(N) / N,
        bounds=[(0, 1)] * N,
        constraints=cons,
        method='SLSQP',
        options={'ftol': 1e-10, 'maxiter': 2000},
    )
    return res.x if res.success else np.ones(N) / N


def frontier(esg_min=None, n_pts=70, rets=None, esg_scores=None):
    if rets is None:
        rets = RETS
    if esg_scores is None:
        esg_scores = ESG

    fv, fr = [], []
    for tr in np.linspace(rets.min() + 0.002, rets.max() - 0.002, n_pts):
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, r=tr: w @ rets - r},
        ]
        if esg_min is not None:
            cons.append({'type': 'ineq', 'fun': lambda w, s=esg_min: w @ esg_scores - s})

        res = minimize(
            lambda w: w @ COV @ w,
            x0=np.ones(N) / N,
            bounds=[(0, 1)] * N,
            constraints=cons,
            method='SLSQP',
            options={'ftol': 1e-10, 'maxiter': 2000},
        )
        if res.success:
            r, v = port_stats(res.x, rets)
            fv.append(v)
            fr.append(r)
    return np.array(fv), np.array(fr)


def compute_score(w, rf=RF, rets=None, esg_scores=None, bonus=0):
    if rets is None:
        rets = RETS
    if esg_scores is None:
        esg_scores = ESG

    r, v = port_stats(w, rets)
    sh = (r - rf) / v if v > 1e-8 else 0
    esg = float(w @ esg_scores)
    s_sharpe = min(max(sh / 1.5, 0), 1) * 50
    s_esg = min(esg / 100, 1) * 30
    total = round(s_sharpe + s_esg + bonus, 1)
    return {
        'return': r,
        'vol': v,
        'sharpe': sh,
        'esg': esg,
        'score_sharpe': s_sharpe,
        'score_esg': s_esg,
        'bonus': bonus,
        'total': total,
    }


def get_event_data(event_label: str):
    rets_shock = RETS.copy()
    esg_shock = ESG.copy()
    desc = ""

    if event_label == "1 — Krach sectoriel Énergie":
        desc = "📉 Krach sectoriel Énergie : TotalEnergies chute de 15%"
        rets_shock[NAMES.index('TotalEnergies')] -= 0.15
    elif event_label == "2 — Scandale ESG Stellantis":
        desc = "💥 Scandale ESG Stellantis : -20% et score ESG effondré"
        rets_shock[NAMES.index('Stellantis')] -= 0.20
        esg_shock[NAMES.index('Stellantis')] = 5
    elif event_label == "3 — Rally Green Tech":
        desc = "🌿 Rally Green Tech : Schneider +15%, Air Liquide +10%"
        rets_shock[NAMES.index('Schneider')] += 0.15
        rets_shock[NAMES.index('Air Liquide')] += 0.10
    else:
        desc = "Aucun événement sélectionné"

    return rets_shock, esg_shock, desc


def weights_from_state(prefix: str):
    vals = np.array([st.session_state.get(f"{prefix}_{n}", 0) for n in NAMES], dtype=float)
    return vals / 100.0, float(vals.sum())


def nonzero_weights_df(w, esg_scores=None):
    if esg_scores is None:
        esg_scores = ESG
    df = pd.DataFrame({
        'Actif': NAMES,
        'Poids (%)': (w * 100).round(1),
        'ESG': esg_scores,
    })
    return df[df['Poids (%)'] > 0]


def make_scatter_assets(rets=None, esg_scores=None, title="Espace Risque / Rendement — Actifs disponibles"):
    if rets is None:
        rets = RETS
    if esg_scores is None:
        esg_scores = ESG
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(VOLS * 100, rets * 100, c=esg_scores, cmap='RdYlGn', vmin=0, vmax=100,
                    s=160, zorder=5, edgecolors='white', lw=1.5)
    plt.colorbar(sc, ax=ax, label='Score ESG')
    for i, name in enumerate(NAMES):
        ax.annotate(name, (VOLS[i] * 100, rets[i] * 100), textcoords='offset points', xytext=(7, 4), fontsize=9)
    ax.axhline(np.mean(rets) * 100, color='gray', ls=':', alpha=0.5, label='Rend. moyen')
    ax.axvline(np.mean(VOLS) * 100, color='gray', ls='--', alpha=0.5, label='Vol. moyenne')
    ax.set_xlabel('Volatilité (%)')
    ax.set_ylabel('Rendement espéré (%)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def make_frontier_fig_acte1():
    fv_ref, fr_ref = frontier(esg_min=None)
    w_opt_ref = optimize(esg_min=None, rf=RF)
    r_ref, v_ref = port_stats(w_opt_ref)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(VOLS * 100, RETS * 100, c=ESG, cmap='RdYlGn', vmin=0, vmax=100,
               s=120, zorder=5, edgecolors='white', lw=1.2)
    for i, name in enumerate(NAMES):
        ax.annotate(name, (VOLS[i] * 100, RETS[i] * 100), textcoords='offset points', xytext=(6, 3), fontsize=8.5)
    ax.plot(fv_ref * 100, fr_ref * 100, '--', color=C_BLUE, lw=2.5, label='Frontière efficiente')
    ax.scatter([v_ref * 100], [r_ref * 100], marker='*', s=300, color=C_GOLD, zorder=6,
               label=f"Max Sharpe = {(r_ref - RF) / v_ref:.3f}")
    x_cml = np.linspace(0, max(VOLS) * 100 * 1.1, 100)
    ax.plot(x_cml, RF * 100 + (r_ref - RF) / v_ref * x_cml, ':', color=C_GOLD, alpha=0.6, label='CML')
    ax.scatter([0], [RF * 100], marker='o', s=80, color='gray', zorder=6, label=f'Rf={RF*100:.1f}%')
    ax.set_xlabel('Volatilité (%)')
    ax.set_ylabel('Rendement espéré (%)')
    ax.set_title('Frontière efficiente — Acte 1', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig, w_opt_ref


def make_frontier_fig_acte2(rets_shock, esg_shock):
    fv_c2, fr_c2 = frontier(esg_min=None, rets=rets_shock, esg_scores=esg_shock)
    fv_e2, fr_e2 = frontier(esg_min=ESG_MIN, rets=rets_shock, esg_scores=esg_shock)
    w_opt_c2 = optimize(esg_min=None, rf=RF, rets=rets_shock, esg_scores=esg_shock)
    w_opt_e2 = optimize(esg_min=ESG_MIN, rf=RF, rets=rets_shock, esg_scores=esg_shock)
    r_c2, v_c2 = port_stats(w_opt_c2, rets_shock)
    r_e2, v_e2 = port_stats(w_opt_e2, rets_shock)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(VOLS * 100, rets_shock * 100, c=esg_shock, cmap='RdYlGn', vmin=0, vmax=100,
               s=120, zorder=5, edgecolors='white', lw=1.2)
    for i, name in enumerate(NAMES):
        ax.annotate(name, (VOLS[i] * 100, rets_shock[i] * 100), textcoords='offset points', xytext=(6, 3), fontsize=8.5)
    ax.plot(fv_c2 * 100, fr_c2 * 100, '--', color=C_BLUE, lw=2, label='Frontière libre')
    ax.plot(fv_e2 * 100, fr_e2 * 100, '-', color=C_GREEN, lw=2.5, label=f'Frontière ESG≥{ESG_MIN}')
    ax.scatter([v_c2 * 100], [r_c2 * 100], marker='*', s=280, color=C_BLUE, zorder=6,
               label=f'Max Sharpe libre ({(r_c2 - RF) / v_c2:.3f})')
    ax.scatter([v_e2 * 100], [r_e2 * 100], marker='*', s=280, color=C_GREEN, zorder=6,
               label=f'Max Sharpe ESG ({(r_e2 - RF) / v_e2:.3f})')
    ax.annotate('', xy=(v_e2 * 100, r_e2 * 100), xytext=(v_c2 * 100, r_c2 * 100),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    mid_x = (v_c2 + v_e2) * 50
    mid_y = (r_c2 + r_e2) * 50
    ax.text(mid_x + 0.2, mid_y, 'coût\nESG', color='orange', fontsize=9, va='center')
    ax.set_xlabel('Volatilité (%)')
    ax.set_ylabel('Rendement (%)')
    ax.set_title('Frontières après événement', fontweight='bold')
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    cats = ['Sharpe', 'Rend. (%)', 'Vol. (%)', 'ESG moyen']
    v_c = [(r_c2 - RF) / v_c2, r_c2 * 100, v_c2 * 100, w_opt_c2 @ esg_shock]
    v_e = [(r_e2 - RF) / v_e2, r_e2 * 100, v_e2 * 100, w_opt_e2 @ esg_shock]
    x = np.arange(len(cats))
    w_bar = 0.35
    b1 = ax2.bar(x - w_bar / 2, v_c, w_bar, label='Libre', color=C_BLUE, alpha=0.85)
    b2 = ax2.bar(x + w_bar / 2, v_e, w_bar, label=f'ESG≥{ESG_MIN}', color=C_GREEN, alpha=0.85)
    for b in list(b1) + list(b2):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                 f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    cout = ((r_c2 - RF) / v_c2 - (r_e2 - RF) / v_e2) / ((r_c2 - RF) / v_c2) * 100
    ax2.set_title(f'Comparaison optimaux\nCoût ESG sur Sharpe : {cout:.1f}%', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.legend()
    ax2.set_ylim(0, max(max(v_c), max(v_e)) * 1.22)

    fig.suptitle('Acte 2 — Impact de la contrainte ESG', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig, w_opt_c2, w_opt_e2, cout


def make_compare_fig(w1, w2, rets_shock, esg_shock, team_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, w, titre, rets_used, esg_used in [
        (axes[0], w1, f'Acte 1 — {team_name}', RETS, ESG),
        (axes[1], w2, f'Acte 2 — {team_name} (post-événement)', rets_shock, esg_shock),
    ]:
        idx = np.where(w > 0.005)[0]
        lbls = [NAMES[i] for i in idx]
        vals = [w[i] * 100 for i in idx]
        esgs = [esg_used[i] for i in idx]
        cols = plt.cm.RdYlGn(np.array(esgs) / 100) if len(esgs) else []
        bars = ax.barh(lbls, vals, color=cols, edgecolor='white', linewidth=1.5)
        for bar, e in zip(bars, esgs):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'ESG {e}', va='center', fontsize=8.5)
        r, v = port_stats(w, rets_used)
        sharpe = (r - RF) / v if v > 1e-8 else 0
        ax.set_xlabel('Poids (%)')
        ax.set_title(f'{titre}\nSharpe={sharpe:.3f} | ESG={w @ esg_used:.1f}', fontweight='bold')
        ax.set_xlim(0, max(vals) * 1.25 if vals else 10)

    fig.suptitle('Évolution de la composition — Acte 1 → Acte 2', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ── État initial ─────────────────────────────────────────────────────────
if 'team_name' not in st.session_state:
    st.session_state.team_name = 'MonEquipe'
if 'event_label' not in st.session_state:
    st.session_state.event_label = '1 — Krach sectoriel Énergie'

for name in NAMES:
    st.session_state.setdefault(f'a1_{name}', 0)
    st.session_state.setdefault(f'a2_{name}', 0)

# ── Header ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background:{C_BLUE};padding:26px;border-radius:12px;color:white;text-align:center;margin-bottom:20px;">
        <h1 style="margin:0;">🏦 GreenAlpha Challenge</h1>
        <h3 style="color:#27AE60;margin:8px 0;">Serious Game — Gestion de Portefeuille & Finance Durable</h3>
        <p style="color:#D7DBDD;margin:0;">M1 Finance — Grenoble IAE — Investissements et Marchés Financiers</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
### 🎯 Votre mission
Vous êtes **gérants juniors** chez *GreenAlpha Asset Management*.
Votre équipe doit construire, défendre et optimiser un portefeuille d'actions européennes — d'abord librement, puis sous contrainte ESG imposée par une nouvelle directive réglementaire.
"""
)

colA, colB, colC = st.columns(3)
colA.info("⚡ Briefing\n\nDécouverte des actifs et des règles")
colB.info("🎯 Acte 1\n\nPortefeuille libre (Markowitz)")
colC.info("🌿 Acte 2\n\nRéallocation sous contrainte ESG")

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres du jeu")
    st.text_input("Nom de l'équipe", key='team_name')
    st.selectbox(
        "Événement révélé par le professeur",
        [
            '1 — Krach sectoriel Énergie',
            '2 — Scandale ESG Stellantis',
            '3 — Rally Green Tech',
        ],
        key='event_label',
    )
    st.markdown("---")
    st.write(f"Taux sans risque : **{RF*100:.1f}%**")
    st.write(f"Seuil ESG Acte 2 : **{ESG_MIN}**")

rets_shock, esg_shock, desc_evt = get_event_data(st.session_state.event_label)

# ── Briefing ─────────────────────────────────────────────────────────────
st.header("⚡ Briefing — Découvrez votre univers d'investissement")
st.dataframe(get_asset_df(), use_container_width=True, hide_index=True)
st.pyplot(make_scatter_assets())

# ── Acte 1 ────────────────────────────────────────────────────────────────
st.header("🎯 Acte 1 — Construisez votre portefeuille libre")
st.markdown(
    "> **Consigne :** aucune contrainte pour l'instant. Votre objectif est de maximiser le ratio de Sharpe."
)

fig1, w_opt_ref = make_frontier_fig_acte1()
st.pyplot(fig1)
r_ref, v_ref = port_stats(w_opt_ref)
st.info(
    f"💡 Portefeuille Max Sharpe de référence : Rend={r_ref*100:.2f}% | Vol={v_ref*100:.2f}% | Sharpe={(r_ref-RF)/v_ref:.3f} | ESG={w_opt_ref@ESG:.1f}"
)

st.subheader("✏️ Saisissez vos poids — Acte 1")
cols1 = st.columns(2)
for i, name in enumerate(NAMES):
    with cols1[i % 2]:
        st.number_input(f"{name} (%)", min_value=0, max_value=100, step=5, key=f'a1_{name}')

w1, total1 = weights_from_state('a1')
st.write(f"**Somme des poids Acte 1 : {total1:.0f}%**")
if abs(total1 - 100) > 0.1:
    st.error("La somme des poids de l'Acte 1 doit être égale à 100%.")
else:
    r1, v1 = port_stats(w1)
    sharpe1 = (r1 - RF) / v1 if v1 > 1e-8 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement", f"{r1*100:.2f}%")
    c2.metric("Volatilité", f"{v1*100:.2f}%")
    c3.metric("Sharpe", f"{sharpe1:.3f}")
    c4.metric("ESG moyen", f"{w1@ESG:.1f}")
    df_w1 = nonzero_weights_df(w1)
    if not df_w1.empty:
        st.dataframe(df_w1, use_container_width=True, hide_index=True)

# ── Événement ────────────────────────────────────────────────────────────
st.header("📰 Événement — Choc de marché")
st.warning(desc_evt)
if abs(total1 - 100) <= 0.1:
    r1_choc, v1_choc = port_stats(w1, rets_shock)
    old_sh = (RETS @ w1 - RF) / np.sqrt(w1 @ COV @ w1) if (w1 @ COV @ w1) > 1e-8 else 0
    new_sh = (r1_choc - RF) / v1_choc if v1_choc > 1e-8 else 0
    cc1, cc2 = st.columns(2)
    cc1.metric("Rendement avant choc", f"{(RETS @ w1)*100:.2f}%")
    cc2.metric("Rendement après choc", f"{r1_choc*100:.2f}%", delta=f"{(r1_choc-RETS@w1)*100:+.2f} pp")
    st.write(f"**Sharpe** : {old_sh:.3f} → {new_sh:.3f}")

# ── Acte 2 ────────────────────────────────────────────────────────────────
st.header("🌿 Acte 2 — Recomposez sous contrainte ESG")
st.markdown(
    "> **Contexte réglementaire :** votre fonds doit désormais afficher un **score ESG moyen ≥ 55** pour conserver son label Article 8."
)

fig2, w_opt_c2, w_opt_e2, cout_esg = make_frontier_fig_acte2(rets_shock, esg_shock)
st.pyplot(fig2)
st.info(f"💡 Coût ESG sur le Sharpe : {cout_esg:.1f}%")

st.subheader("✏️ Saisissez vos poids — Acte 2")
cols2 = st.columns(2)
for i, name in enumerate(NAMES):
    with cols2[i % 2]:
        st.number_input(f"{name} (%) ", min_value=0, max_value=100, step=5, key=f'a2_{name}')

w2, total2 = weights_from_state('a2')
esg2_val = float(w2 @ esg_shock)
st.write(f"**Somme des poids Acte 2 : {total2:.0f}%**")

if abs(total2 - 100) > 0.1:
    st.error("La somme des poids de l'Acte 2 doit être égale à 100%.")
elif esg2_val < ESG_MIN:
    st.error(f"Contrainte SFDR non respectée : score ESG = {esg2_val:.1f} < {ESG_MIN}")
    r2, v2 = port_stats(w2, rets_shock)
    sharpe2 = (r2 - RF) / v2 if v2 > 1e-8 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement", f"{r2*100:.2f}%")
    c2.metric("Volatilité", f"{v2*100:.2f}%")
    c3.metric("Sharpe", f"{sharpe2:.3f}")
    c4.metric("ESG moyen", f"{esg2_val:.1f}")
else:
    st.success(f"Contrainte ESG respectée : {esg2_val:.1f} ≥ {ESG_MIN}")
    r2, v2 = port_stats(w2, rets_shock)
    sharpe2 = (r2 - RF) / v2 if v2 > 1e-8 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement", f"{r2*100:.2f}%")
    c2.metric("Volatilité", f"{v2*100:.2f}%")
    c3.metric("Sharpe", f"{sharpe2:.3f}")
    c4.metric("ESG moyen", f"{esg2_val:.1f}")
    df_w2 = nonzero_weights_df(w2, esg_scores=esg_shock)
    if not df_w2.empty:
        st.dataframe(df_w2, use_container_width=True, hide_index=True)

# ── Score final ──────────────────────────────────────────────────────────
st.header("📊 Score final")
if abs(total1 - 100) <= 0.1 and abs(total2 - 100) <= 0.1:
    s1 = compute_score(w1, rf=RF, rets=RETS, esg_scores=ESG, bonus=0)
    penalite = -15 if esg2_val < ESG_MIN else 0
    bonus_conformite = 10 if esg2_val >= 65 else 0
    s2 = compute_score(w2, rf=RF, rets=rets_shock, esg_scores=esg_shock, bonus=penalite + bonus_conformite)
    score_total = round(s1['total'] + s2['total'], 1)

    left, right = st.columns(2)
    with left:
        st.subheader("Acte 1")
        st.write(f"Sharpe : **{s1['sharpe']:.3f}** → {s1['score_sharpe']:.1f}/50")
        st.write(f"ESG : **{s1['esg']:.1f}** → {s1['score_esg']:.1f}/30")
        st.write(f"Total Acte 1 : **{s1['total']:.1f}/80**")
    with right:
        st.subheader("Acte 2")
        st.write(f"Sharpe : **{s2['sharpe']:.3f}** → {s2['score_sharpe']:.1f}/50")
        st.write(f"ESG : **{s2['esg']:.1f}** → {s2['score_esg']:.1f}/30")
        st.write(f"Bonus / pénalité : **{s2['bonus']:+.0f}**")
        st.write(f"Total Acte 2 : **{s2['total']:.1f}/80**")

    st.success(f"🏆 Score total de {st.session_state.team_name} : {score_total:.1f} / 160")
    st.progress(min(max(score_total / 160, 0.0), 1.0))
else:
    st.info("Complète d'abord l'Acte 1 et l'Acte 2 avec une somme de 100% pour afficher le score final.")

# ── Pitch ────────────────────────────────────────────────────────────────
st.header("🎤 Pitch — Défendez vos choix")
if abs(total1 - 100) <= 0.1 and abs(total2 - 100) <= 0.1:
    st.pyplot(make_compare_fig(w1, w2, rets_shock, esg_shock, st.session_state.team_name))

st.markdown(
    """
**Questions pour le pitch**
1. Quels actifs avez-vous renforcés ou réduits entre l'Acte 1 et l'Acte 2 ?
2. Quel est votre coût ESG en perte de Sharpe ? Est-ce justifié ?
3. Si le seuil ESG passait à 70, que feriez-vous ?
"""
)

# ── Correction ───────────────────────────────────────────────────────────
st.header("📊 Correction — Portefeuille optimal")
r_e, v_e = port_stats(w_opt_e2, rets_shock)
r_libre, v_libre = port_stats(w_opt_c2, rets_shock)
cout_esg_corr = ((r_libre - RF) / v_libre - (r_e - RF) / v_e) / ((r_libre - RF) / v_libre) * 100

st.write(f"**Rendement optimal ESG** : {r_e*100:.2f}%")
st.write(f"**Volatilité optimale ESG** : {v_e*100:.2f}%")
st.write(f"**Sharpe optimal ESG** : {(r_e-RF)/v_e:.3f}")
st.write(f"**ESG moyen optimal** : {w_opt_e2 @ esg_shock:.1f}")

if abs(total2 - 100) <= 0.1:
    df_opt = pd.DataFrame({
        'Actif': NAMES,
        'Poids optimal (%)': (w_opt_e2 * 100).round(1),
        'Vos poids Acte 2 (%)': (w2 * 100).round(1),
        'ESG': esg_shock,
    })
    df_opt = df_opt[(df_opt['Poids optimal (%)'] > 0.5) | (df_opt['Vos poids Acte 2 (%)'] > 0.5)]
    df_opt['Écart (pp)'] = (df_opt['Vos poids Acte 2 (%)'] - df_opt['Poids optimal (%)']).round(1)
    st.dataframe(df_opt, use_container_width=True, hide_index=True)

st.markdown(
    f"""
**Leçons clés**
- Coût ESG sur Sharpe : **{cout_esg_corr:.1f}%**
- ESG moyen optimal : **{w_opt_e2 @ esg_shock:.1f}** pour un seuil imposé à **{ESG_MIN}**
- Diversification : elle protège davantage qu'une allocation trop concentrée
"""
)

st.caption("Références : Markowitz (1952) · Pedersen et al. (2021) · Règlement SFDR (UE) 2019/2088")
