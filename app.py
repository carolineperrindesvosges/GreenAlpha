"""
GreenAlpha Challenge — Serious Game M1 Finance, Grenoble IAE
Cours : Investissements et marchés financiers (24h)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

# ── CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenAlpha Challenge",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette sage + argile (cohérente avec les slides)
C_SAGE_DEEP = '#4A6655'
C_SAGE      = '#6B8676'
C_SAGE_MID  = '#94B0A1'
C_SAGE_BG   = '#EFF4F0'
C_CLAY_DEEP = '#8E6645'
C_CLAY      = '#B08361'
C_CLAY_MID  = '#C9A085'
C_CLAY_BG   = '#F4E6D3'
C_TEXT      = '#2D332F'
C_MUTED     = '#8A938D'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#FAFAFA',
    'axes.grid':        True,
    'grid.alpha':       0.20,
    'grid.color':       '#CCCCCC',
    'font.family':      'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.edgecolor':    '#BBBBBB',
})

# ── UNIVERS ───────────────────────────────────────────────────────────────
ASSETS = {
    'TotalEnergies': {'ret': 0.082, 'vol': 0.221, 'esg': 32, 'sector': 'Énergie'},
    'Schneider':     {'ret': 0.134, 'vol': 0.198, 'esg': 78, 'sector': 'Industrie'},
    'Air Liquide':   {'ret': 0.096, 'vol': 0.172, 'esg': 65, 'sector': 'Chimie'},
    'BNP Paribas':   {'ret': 0.071, 'vol': 0.263, 'esg': 44, 'sector': 'Finance'},
    'Danone':        {'ret': 0.058, 'vol': 0.189, 'esg': 71, 'sector': 'Conso.'},
    'Stellantis':    {'ret': 0.105, 'vol': 0.298, 'esg': 29, 'sector': 'Auto'},
    "L'Oréal":       {'ret': 0.118, 'vol': 0.181, 'esg': 82, 'sector': 'Luxe'},
    'Vinci':         {'ret': 0.089, 'vol': 0.207, 'esg': 56, 'sector': 'Infra'},
    'Sanofi':        {'ret': 0.076, 'vol': 0.194, 'esg': 68, 'sector': 'Santé'},
    'Engie':         {'ret': 0.063, 'vol': 0.241, 'esg': 51, 'sector': 'Utilities'},
}
NAMES = list(ASSETS.keys())
N     = len(NAMES)
RETS  = np.array([ASSETS[a]['ret'] for a in NAMES])
VOLS  = np.array([ASSETS[a]['vol'] for a in NAMES])
ESG   = np.array([ASSETS[a]['esg'] for a in NAMES])
RF      = 0.025
ESG_MIN = 55

# Matrice de covariance stable (semence fixée)
def _build_cov():
    rng = np.random.default_rng(42)
    A   = rng.standard_normal((N, N)) * 0.15
    C   = np.eye(N) + A + A.T
    C  /= np.abs(C).max(axis=1, keepdims=True)
    np.fill_diagonal(C, 1.0)
    C   = (C + C.T) / 2
    ev  = np.linalg.eigvals(C).real
    if ev.min() < 0:
        C += (-ev.min() + 0.01) * np.eye(N)
    C  /= np.sqrt(np.outer(np.diag(C), np.diag(C)))
    np.fill_diagonal(C, 1.0)
    return np.outer(VOLS, VOLS) * C

COV = _build_cov()

# ── FONCTIONS COEUR ───────────────────────────────────────────────────────
def port_stats(w, rets=RETS):
    return float(w @ rets), float(np.sqrt(w @ COV @ w))

def sharpe(w, rets=RETS, rf=RF):
    r, v = port_stats(w, rets)
    return (r - rf) / v if v > 1e-8 else 0.0

@st.cache_data(show_spinner=False)
def optimize(esg_min=None, rf=RF, rets_tuple=None, esg_tuple=None):
    rets = np.array(rets_tuple) if rets_tuple is not None else RETS
    esg  = np.array(esg_tuple)  if esg_tuple  is not None else ESG
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if esg_min is not None:
        cons.append({'type': 'ineq', 'fun': lambda w, s=esg_min: w @ esg - s})

    def neg_sharpe(w):
        r, v = port_stats(w, rets)
        return -(r - rf) / v if v > 1e-8 else 0.0

    res = minimize(
        neg_sharpe, x0=np.ones(N) / N,
        bounds=[(0, 1)] * N, constraints=cons,
        method='SLSQP', options={'ftol': 1e-10, 'maxiter': 2000},
    )
    return res.x if res.success else np.ones(N) / N

@st.cache_data(show_spinner=False)
def frontier(esg_min=None, n_pts=70, rets_tuple=None, esg_tuple=None):
    rets = np.array(rets_tuple) if rets_tuple is not None else RETS
    esg  = np.array(esg_tuple)  if esg_tuple  is not None else ESG
    fv, fr = [], []
    for tr in np.linspace(rets.min() + 0.002, rets.max() - 0.002, n_pts):
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, r=tr: w @ rets - r},
        ]
        if esg_min is not None:
            cons.append({'type': 'ineq', 'fun': lambda w, s=esg_min: w @ esg - s})
        res = minimize(
            lambda w: w @ COV @ w, x0=np.ones(N) / N,
            bounds=[(0, 1)] * N, constraints=cons,
            method='SLSQP', options={'ftol': 1e-10, 'maxiter': 2000},
        )
        if res.success:
            r, v = port_stats(res.x, rets)
            fv.append(v); fr.append(r)
    return np.array(fv), np.array(fr)

def get_event_data(event_label):
    rets_shock = RETS.copy()
    esg_shock  = ESG.copy()
    if event_label.startswith("1"):
        desc = "📉 Krach sectoriel Énergie : TotalEnergies −15%"
        rets_shock[NAMES.index('TotalEnergies')] -= 0.15
    elif event_label.startswith("2"):
        desc = "💥 Scandale ESG Stellantis : −20% et score ESG effondré"
        rets_shock[NAMES.index('Stellantis')] -= 0.20
        esg_shock[NAMES.index('Stellantis')]   = 5
    elif event_label.startswith("3"):
        desc = "🌿 Rally Green Tech : Schneider +15%, Air Liquide +10%"
        rets_shock[NAMES.index('Schneider')]   += 0.15
        rets_shock[NAMES.index('Air Liquide')] += 0.10
    else:
        desc = "Aucun événement"
    return rets_shock, esg_shock, desc

def compute_score(w, rets, esg_scores, rf=RF, bonus=0.0):
    r, v = port_stats(w, rets)
    sh   = (r - rf) / v if v > 1e-8 else 0
    esg  = float(w @ esg_scores)
    s_sh = min(max(sh / 1.5, 0), 1) * 50
    s_es = min(esg / 100, 1) * 30
    return {
        'return': r, 'vol': v, 'sharpe': sh, 'esg': esg,
        'score_sharpe': s_sh, 'score_esg': s_es,
        'bonus': bonus, 'total': round(s_sh + s_es + bonus, 1),
    }

def weights_from_state(prefix):
    vals = np.array([st.session_state.get(f"{prefix}_{n}", 0) for n in NAMES], dtype=float)
    return vals / 100.0, float(vals.sum())

def set_weights_from_array(prefix, w_array):
    pct = (w_array * 100).round(0).astype(int)
    diff = 100 - pct.sum()
    if diff != 0:
        idx = int(np.argmax(pct))
        pct[idx] += diff
    for i, name in enumerate(NAMES):
        st.session_state[f"{prefix}_{name}"] = int(pct[i])

# ── ÉTAT INITIAL ──────────────────────────────────────────────────────────
def init_state():
    if 'team_name' not in st.session_state:
        st.session_state.team_name = 'MonEquipe'
    if 'event_label' not in st.session_state:
        st.session_state.event_label = '1 — Krach sectoriel Énergie'
    for name in NAMES:
        st.session_state.setdefault(f'a1_{name}', 0)
        st.session_state.setdefault(f'a2_{name}', 0)

init_state()

# ── HEADER ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,{C_SAGE_DEEP},{C_SAGE});
                padding:24px;border-radius:10px;color:white;text-align:center;
                margin-bottom:20px;">
        <h1 style="margin:0;font-weight:600;">🏦 GreenAlpha Challenge</h1>
        <p style="color:{C_CLAY_MID};margin:6px 0 0 0;font-size:1.05em;">
          Serious Game · Gestion de portefeuille \\& Finance durable
        </p>
        <p style="color:#D7DBDD;margin:4px 0 0 0;font-size:0.85em;">
          M1 Finance · Grenoble IAE · Cours « Investissements et marchés financiers »
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    st.text_input("Nom de l'équipe", key='team_name')
    st.selectbox(
        "Événement révélé par le professeur",
        ['1 — Krach sectoriel Énergie',
         '2 — Scandale ESG Stellantis',
         '3 — Rally Green Tech'],
        key='event_label',
    )
    st.markdown("---")
    st.caption(f"Taux sans risque : **{RF*100:.1f}%**")
    st.caption(f"Seuil ESG Acte 2 : **{ESG_MIN}**")
    st.caption("Univers : 10 actions Euronext")
    st.markdown("---")
    if st.button("🔄 Réinitialiser tous les poids", use_container_width=True):
        for name in NAMES:
            st.session_state[f'a1_{name}'] = 0
            st.session_state[f'a2_{name}'] = 0
        st.rerun()

rets_shock, esg_shock, desc_evt = get_event_data(st.session_state.event_label)

# ── ONGLETS ───────────────────────────────────────────────────────────────
tab_brief, tab_a1, tab_evt, tab_a2, tab_score, tab_pitch = st.tabs(
    ["⚡ Briefing", "🎯 Acte 1", "📰 Événement", "🌿 Acte 2", "📊 Score", "🎤 Pitch & Correction"]
)

# ── BRIEFING ──────────────────────────────────────────────────────────────
with tab_brief:
    st.markdown("### Mission")
    st.markdown(
        f"""
        Vous êtes **gérants juniors** chez *GreenAlpha Asset Management*.  
        Votre équipe **{st.session_state.team_name}** doit construire, défendre et
        optimiser un portefeuille d'actions européennes — d'abord librement,
        puis sous contrainte ESG imposée par une nouvelle directive réglementaire (SFDR Art. 8).
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.info("⚡ **Briefing**\n\nDécouverte des actifs et des règles")
    c2.info("🎯 **Acte 1**\n\nPortefeuille libre (Markowitz)")
    c3.info("🌿 **Acte 2**\n\nRéallocation sous contrainte ESG")

    st.markdown("### Univers d'investissement")
    df_assets = pd.DataFrame({
        'Actif': NAMES,
        'Secteur': [ASSETS[a]['sector'] for a in NAMES],
        'Rendement (%)':  (RETS * 100).round(1),
        'Volatilité (%)': (VOLS * 100).round(1),
        'Score ESG /100': ESG,
    })

    def color_esg(val):
        if val >= 65: return f'background-color:{C_SAGE_BG};color:{C_SAGE_DEEP};font-weight:600'
        if val >= 45: return 'background-color:#FFF8E5;color:#856404'
        return f'background-color:{C_CLAY_BG};color:{C_CLAY_DEEP}'

    st.dataframe(
        df_assets.style.applymap(color_esg, subset=['Score ESG /100']),
        use_container_width=True, hide_index=True,
    )

    # Scatter risque/rendement coloré ESG
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sc = ax.scatter(VOLS * 100, RETS * 100, c=ESG, cmap='RdYlGn',
                    vmin=20, vmax=90, s=180, zorder=5,
                    edgecolors='white', lw=1.8)
    plt.colorbar(sc, ax=ax, label='Score ESG')
    for i, name in enumerate(NAMES):
        ax.annotate(name, (VOLS[i]*100, RETS[i]*100),
                    textcoords='offset points', xytext=(8, 5), fontsize=9)
    ax.axhline(np.mean(RETS)*100, color=C_MUTED, ls=':', alpha=0.5, label='Rdt. moyen')
    ax.axvline(np.mean(VOLS)*100, color=C_MUTED, ls='--', alpha=0.5, label='Vol. moyenne')
    ax.set_xlabel('Volatilité (%)'); ax.set_ylabel('Rendement espéré (%)')
    ax.set_title("Espace risque / rendement — couleur = score ESG",
                 fontweight='bold', color=C_SAGE_DEEP)
    ax.legend(fontsize=9); fig.tight_layout()
    st.pyplot(fig)

    with st.expander("ℹ️ Conseils pour bien démarrer"):
        st.markdown("""
        - Regardez d'abord les **actions vertes mais peu rentables** vs les **brunes mais lucratives** : c'est l'arbitrage central du jeu.
        - Une **volatilité élevée** ne signifie pas mauvais actif : la **diversification** peut le rendre utile.
        - Le **score ESG d'un portefeuille** est la moyenne *pondérée par les poids*.
        """)

# ── ACTE 1 ────────────────────────────────────────────────────────────────
with tab_a1:
    st.markdown("### 🎯 Acte 1 — Portefeuille libre")
    st.markdown(
        "> **Consigne :** aucune contrainte ESG. Maximisez le **ratio de Sharpe**. "
        "Vous disposez de 100% à allouer entre les 10 actifs."
    )

    # Frontière efficiente Acte 1
    fv_ref, fr_ref = frontier()
    w_opt_ref = optimize()
    r_ref, v_ref = port_stats(w_opt_ref)
    sharpe_ref = (r_ref - RF) / v_ref

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(VOLS*100, RETS*100, c=ESG, cmap='RdYlGn', vmin=20, vmax=90,
               s=120, zorder=5, edgecolors='white', lw=1.3)
    for i, name in enumerate(NAMES):
        ax.annotate(name, (VOLS[i]*100, RETS[i]*100),
                    textcoords='offset points', xytext=(6, 4), fontsize=8.5)
    ax.plot(fv_ref*100, fr_ref*100, '--', color=C_SAGE_DEEP, lw=2.2,
            label='Frontière efficiente')
    ax.scatter([v_ref*100], [r_ref*100], marker='*', s=320,
               color=C_CLAY, zorder=6, edgecolor='white', lw=1,
               label=f"Max Sharpe = {sharpe_ref:.3f}")
    x_cml = np.linspace(0, max(VOLS)*100*1.1, 100)
    ax.plot(x_cml, RF*100 + sharpe_ref*x_cml, ':',
            color=C_CLAY_MID, alpha=0.7, label='CML')
    ax.set_xlabel('Volatilité (%)'); ax.set_ylabel('Rendement espéré (%)')
    ax.set_title('Frontière efficiente — Acte 1', fontweight='bold', color=C_SAGE_DEEP)
    ax.legend(fontsize=9); fig.tight_layout()
    st.pyplot(fig)

    st.info(
        f"💡 **Optimum théorique** : Rdt={r_ref*100:.2f}% · Vol={v_ref*100:.2f}% · "
        f"Sharpe={sharpe_ref:.3f} · ESG={w_opt_ref@ESG:.1f}"
    )

    # Boutons d'aide
    bcol1, bcol2, bcol3 = st.columns(3)
    if bcol1.button("⚖️ Équipondération (10% chacun)", use_container_width=True):
        set_weights_from_array('a1', np.ones(N) / N); st.rerun()
    if bcol2.button("🎯 Copier l'optimum théorique", use_container_width=True):
        set_weights_from_array('a1', w_opt_ref); st.rerun()
    if bcol3.button("🧹 Effacer Acte 1", use_container_width=True):
        for name in NAMES: st.session_state[f'a1_{name}'] = 0
        st.rerun()

    st.markdown("#### ✏️ Saisissez vos poids (en %)")
    cols = st.columns(2)
    for i, name in enumerate(NAMES):
        with cols[i % 2]:
            st.number_input(name, min_value=0, max_value=100, step=5, key=f'a1_{name}')

    w1, total1 = weights_from_state('a1')
    st.progress(min(total1 / 100, 1.0), text=f"Somme des poids : {total1:.0f}% / 100%")

    if abs(total1 - 100) > 0.1:
        st.error("⚠️ La somme des poids doit être exactement égale à 100%.")
    else:
        r1, v1 = port_stats(w1)
        sh1 = (r1 - RF) / v1 if v1 > 1e-8 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rendement",  f"{r1*100:.2f}%")
        c2.metric("Volatilité", f"{v1*100:.2f}%")
        c3.metric("Sharpe",     f"{sh1:.3f}",
                  delta=f"{(sh1 - sharpe_ref):+.3f} vs optimum",
                  delta_color="off")
        c4.metric("ESG moyen",  f"{w1@ESG:.1f}")

        df_w1 = pd.DataFrame({
            'Actif': NAMES, 'Poids (%)': (w1*100).round(1), 'ESG': ESG,
        })
        st.dataframe(df_w1[df_w1['Poids (%)'] > 0],
                     use_container_width=True, hide_index=True)

# ── ÉVÉNEMENT ─────────────────────────────────────────────────────────────
with tab_evt:
    st.markdown("### 📰 Événement de marché")
    st.warning(desc_evt)

    w1, total1 = weights_from_state('a1')
    if abs(total1 - 100) > 0.1:
        st.info("Complétez d'abord votre Acte 1 pour visualiser l'impact.")
    else:
        r_av, v_av = port_stats(w1, RETS)
        r_ap, v_ap = port_stats(w1, rets_shock)
        sh_av = (r_av - RF) / v_av
        sh_ap = (r_ap - RF) / v_ap

        c1, c2 = st.columns(2)
        c1.metric("Rendement avant choc", f"{r_av*100:.2f}%")
        c2.metric("Rendement après choc", f"{r_ap*100:.2f}%",
                  delta=f"{(r_ap - r_av)*100:+.2f} pp")
        c3, c4 = st.columns(2)
        c3.metric("Sharpe avant choc", f"{sh_av:.3f}")
        c4.metric("Sharpe après choc", f"{sh_ap:.3f}",
                  delta=f"{sh_ap - sh_av:+.3f}")

        st.markdown("#### Impact actif par actif")
        df_imp = pd.DataFrame({
            'Actif': NAMES,
            'Poids (%)':       (w1*100).round(1),
            'Rdt avant (%)':   (RETS*100).round(1),
            'Rdt après (%)':   (rets_shock*100).round(1),
            'Δ Rdt (pp)':      ((rets_shock - RETS)*100).round(1),
            'ESG avant':       ESG,
            'ESG après':       esg_shock,
        })
        st.dataframe(df_imp[df_imp['Poids (%)'] > 0],
                     use_container_width=True, hide_index=True)

# ── ACTE 2 ────────────────────────────────────────────────────────────────
with tab_a2:
    st.markdown("### 🌿 Acte 2 — Recomposition sous contrainte ESG")
    st.markdown(
        f"> **Contexte réglementaire :** votre fonds doit afficher un **score ESG moyen ≥ {ESG_MIN}** "
        "pour conserver son label SFDR Article 8."
    )

    # Frontières post-événement
    fv_libre,  fr_libre  = frontier(rets_tuple=tuple(rets_shock), esg_tuple=tuple(esg_shock))
    fv_esg,    fr_esg    = frontier(esg_min=ESG_MIN, rets_tuple=tuple(rets_shock), esg_tuple=tuple(esg_shock))
    w_opt_libre = optimize(rets_tuple=tuple(rets_shock), esg_tuple=tuple(esg_shock))
    w_opt_esg   = optimize(esg_min=ESG_MIN, rets_tuple=tuple(rets_shock), esg_tuple=tuple(esg_shock))
    r_libre, v_libre = port_stats(w_opt_libre, rets_shock)
    r_esg,   v_esg   = port_stats(w_opt_esg,   rets_shock)
    sh_libre = (r_libre - RF) / v_libre
    sh_esg   = (r_esg   - RF) / v_esg
    cout_esg_pct = (sh_libre - sh_esg) / sh_libre * 100 if sh_libre > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(VOLS*100, rets_shock*100, c=esg_shock, cmap='RdYlGn',
               vmin=20, vmax=90, s=120, zorder=5, edgecolors='white', lw=1.3)
    for i, name in enumerate(NAMES):
        ax.annotate(name, (VOLS[i]*100, rets_shock[i]*100),
                    textcoords='offset points', xytext=(6, 3), fontsize=8.5)
    ax.plot(fv_libre*100, fr_libre*100, '--', color=C_SAGE_DEEP, lw=1.8,
            label='Frontière libre')
    ax.plot(fv_esg*100,   fr_esg*100,   '-',  color=C_CLAY,      lw=2.4,
            label=f'Frontière ESG ≥ {ESG_MIN}')
    ax.scatter([v_libre*100], [r_libre*100], marker='*', s=280,
               color=C_SAGE_DEEP, zorder=6, edgecolor='white', lw=1,
               label=f'Max Sharpe libre ({sh_libre:.3f})')
    ax.scatter([v_esg*100],   [r_esg*100],   marker='*', s=280,
               color=C_CLAY,      zorder=6, edgecolor='white', lw=1,
               label=f'Max Sharpe ESG ({sh_esg:.3f})')
    ax.set_xlabel('Volatilité (%)'); ax.set_ylabel('Rendement (%)')
    ax.set_title('Frontières après événement', fontweight='bold', color=C_SAGE_DEEP)
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    cats = ['Sharpe', 'Rdt (%)', 'Vol (%)', 'ESG']
    v_l = [sh_libre, r_libre*100, v_libre*100, w_opt_libre @ esg_shock]
    v_e = [sh_esg,   r_esg*100,   v_esg*100,   w_opt_esg   @ esg_shock]
    x = np.arange(len(cats)); wb = 0.36
    b1 = ax2.bar(x - wb/2, v_l, wb, label='Libre',           color=C_SAGE_DEEP, alpha=0.85)
    b2 = ax2.bar(x + wb/2, v_e, wb, label=f'ESG ≥ {ESG_MIN}', color=C_CLAY,      alpha=0.85)
    for b in list(b1) + list(b2):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15,
                 f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=8.5)
    ax2.set_xticks(x); ax2.set_xticklabels(cats); ax2.legend()
    ax2.set_ylim(0, max(max(v_l), max(v_e)) * 1.22)
    ax2.set_title(f'Comparaison\nCoût ESG sur Sharpe : {cout_esg_pct:.1f}%',
                  fontweight='bold', color=C_SAGE_DEEP)
    fig.tight_layout()
    st.pyplot(fig)

    st.info(f"💡 **Coût ESG** : la contrainte fait perdre {cout_esg_pct:.1f}% de ratio de Sharpe.")

    # Boutons d'aide Acte 2
    bcol1, bcol2, bcol3 = st.columns(3)
    if bcol1.button("⚖️ Équipondération", key='a2_eq', use_container_width=True):
        set_weights_from_array('a2', np.ones(N) / N); st.rerun()
    if bcol2.button("🎯 Copier l'optimum ESG", key='a2_opt', use_container_width=True):
        set_weights_from_array('a2', w_opt_esg); st.rerun()
    if bcol3.button("🧹 Effacer Acte 2", key='a2_clear', use_container_width=True):
        for name in NAMES: st.session_state[f'a2_{name}'] = 0
        st.rerun()

    st.markdown("#### ✏️ Saisissez vos poids — Acte 2 (en %)")
    cols = st.columns(2)
    for i, name in enumerate(NAMES):
        with cols[i % 2]:
            st.number_input(f"{name} ", min_value=0, max_value=100, step=5, key=f'a2_{name}')

    w2, total2 = weights_from_state('a2')
    esg2_val = float(w2 @ esg_shock)
    st.progress(min(total2 / 100, 1.0), text=f"Somme des poids : {total2:.0f}% / 100%")

    if abs(total2 - 100) > 0.1:
        st.error("⚠️ La somme des poids doit être exactement égale à 100%.")
    else:
        r2, v2 = port_stats(w2, rets_shock)
        sh2 = (r2 - RF) / v2 if v2 > 1e-8 else 0
        if esg2_val < ESG_MIN:
            st.error(f"❌ Contrainte SFDR non respectée : ESG = {esg2_val:.1f} < {ESG_MIN}")
        else:
            st.success(f"✅ Contrainte SFDR respectée : ESG = {esg2_val:.1f} ≥ {ESG_MIN}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rendement",  f"{r2*100:.2f}%")
        c2.metric("Volatilité", f"{v2*100:.2f}%")
        c3.metric("Sharpe",     f"{sh2:.3f}",
                  delta=f"{(sh2 - sh_esg):+.3f} vs optimum ESG",
                  delta_color="off")
        c4.metric("ESG moyen",  f"{esg2_val:.1f}")

        df_w2 = pd.DataFrame({
            'Actif': NAMES, 'Poids (%)': (w2*100).round(1), 'ESG': esg_shock,
        })
        st.dataframe(df_w2[df_w2['Poids (%)'] > 0],
                     use_container_width=True, hide_index=True)

# ── SCORE ─────────────────────────────────────────────────────────────────
with tab_score:
    st.markdown("### 📊 Score final")

    with st.expander("🧮 Comment est calculé le score ?"):
        st.markdown(f"""
        Pour **chaque acte** (sur 80 points) :
        - **Sharpe** : `min(Sharpe / 1.5, 1) × 50` → max **50 pts**
        - **ESG**    : `min(ESG / 100, 1) × 30` → max **30 pts**

        Pour **l'acte 2 uniquement** :
        - 🎁 Bonus de **+10 pts** si ESG ≥ 65 (au-delà de la contrainte)
        - ❌ Pénalité de **−15 pts** si ESG < {ESG_MIN} (contrainte SFDR violée)

        **Score total** : Acte 1 + Acte 2, sur **160 points**.
        """)

    w1, total1 = weights_from_state('a1')
    w2, total2 = weights_from_state('a2')
    esg2_val = float(w2 @ esg_shock)

    if abs(total1 - 100) > 0.1 or abs(total2 - 100) > 0.1:
        st.info("⏳ Complétez Acte 1 et Acte 2 (somme = 100%) pour afficher le score.")
    else:
        s1 = compute_score(w1, RETS, ESG)
        bonus = (-15 if esg2_val < ESG_MIN else 0) + (10 if esg2_val >= 65 else 0)
        s2 = compute_score(w2, rets_shock, esg_shock, bonus=bonus)
        score_total = round(s1['total'] + s2['total'], 1)

        l, r = st.columns(2)
        with l:
            st.markdown(f"#### 🎯 Acte 1 — {s1['total']:.1f} / 80")
            st.write(f"Sharpe : **{s1['sharpe']:.3f}** → {s1['score_sharpe']:.1f}/50")
            st.write(f"ESG : **{s1['esg']:.1f}** → {s1['score_esg']:.1f}/30")
        with r:
            st.markdown(f"#### 🌿 Acte 2 — {s2['total']:.1f} / 80")
            st.write(f"Sharpe : **{s2['sharpe']:.3f}** → {s2['score_sharpe']:.1f}/50")
            st.write(f"ESG : **{s2['esg']:.1f}** → {s2['score_esg']:.1f}/30")
            if s2['bonus'] != 0:
                emoji = "🎁" if s2['bonus'] > 0 else "❌"
                st.write(f"{emoji} Bonus / pénalité : **{s2['bonus']:+.0f}**")

        st.markdown("---")
        st.success(f"🏆 **{st.session_state.team_name} : {score_total:.1f} / 160**")
        st.progress(min(max(score_total / 160, 0.0), 1.0))

# ── PITCH & CORRECTION ────────────────────────────────────────────────────
with tab_pitch:
    st.markdown("### 🎤 Pitch — Défendez vos choix (10 min)")

    w1, total1 = weights_from_state('a1')
    w2, total2 = weights_from_state('a2')

    if abs(total1 - 100) <= 0.1 and abs(total2 - 100) <= 0.1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, w, titre, rets_used, esg_used in [
            (axes[0], w1, f'Acte 1 — {st.session_state.team_name}',  RETS,       ESG),
            (axes[1], w2, f'Acte 2 — {st.session_state.team_name}',  rets_shock, esg_shock),
        ]:
            idx = np.where(w > 0.005)[0]
            lbls = [NAMES[i] for i in idx]
            vals = [w[i]*100 for i in idx]
            esgs = [esg_used[i] for i in idx]
            cols = plt.cm.RdYlGn(np.array(esgs) / 100) if len(esgs) else []
            bars = ax.barh(lbls, vals, color=cols, edgecolor='white', linewidth=1.5)
            for bar, e in zip(bars, esgs):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        f'ESG {e}', va='center', fontsize=8.5)
            r, v = port_stats(w, rets_used)
            sh = (r - RF) / v if v > 1e-8 else 0
            ax.set_xlabel('Poids (%)')
            ax.set_title(f'{titre}\nSharpe={sh:.3f} · ESG={w@esg_used:.1f}',
                         fontweight='bold', color=C_SAGE_DEEP)
            ax.set_xlim(0, max(vals)*1.25 if vals else 10)
        fig.suptitle('Évolution Acte 1 → Acte 2',
                     fontsize=12, fontweight='bold', color=C_SAGE_DEEP, y=1.02)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown(
        """
        #### Questions pour le pitch
        1. Quels actifs avez-vous **renforcés ou réduits** entre l'Acte 1 et l'Acte 2 ?
        2. Quel est votre **coût ESG** en perte de Sharpe ? Est-ce justifié ?
        3. Si le **seuil ESG passait à 70**, que feriez-vous ?
        4. Comment votre allocation se compare-t-elle à un fonds **équipondéré** ?
        """
    )

    st.markdown("---")
    st.markdown("### 📊 Correction — Portefeuille optimal sous contrainte ESG")

    w_opt_esg = optimize(esg_min=ESG_MIN,
                         rets_tuple=tuple(rets_shock),
                         esg_tuple=tuple(esg_shock))
    w_opt_libre = optimize(rets_tuple=tuple(rets_shock),
                           esg_tuple=tuple(esg_shock))
    r_e, v_e = port_stats(w_opt_esg, rets_shock)
    r_l, v_l = port_stats(w_opt_libre, rets_shock)
    sh_e = (r_e - RF) / v_e
    sh_l = (r_l - RF) / v_l
    cout = (sh_l - sh_e) / sh_l * 100 if sh_l > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement",  f"{r_e*100:.2f}%")
    c2.metric("Volatilité", f"{v_e*100:.2f}%")
    c3.metric("Sharpe",     f"{sh_e:.3f}")
    c4.metric("ESG moyen",  f"{w_opt_esg @ esg_shock:.1f}")

    if abs(total2 - 100) <= 0.1:
        df_opt = pd.DataFrame({
            'Actif': NAMES,
            'Poids optimal (%)':   (w_opt_esg * 100).round(1),
            'Vos poids Acte 2 (%)': (w2 * 100).round(1),
            'ESG': esg_shock,
        })
        df_opt = df_opt[(df_opt['Poids optimal (%)'] > 0.5) | (df_opt['Vos poids Acte 2 (%)'] > 0.5)]
        df_opt['Écart (pp)'] = (df_opt['Vos poids Acte 2 (%)'] - df_opt['Poids optimal (%)']).round(1)
        st.dataframe(df_opt, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        **Leçons clés du jeu**
        - Coût ESG sur le Sharpe : **{cout:.1f}%**
        - ESG moyen optimal sous contrainte : **{w_opt_esg @ esg_shock:.1f}** (seuil imposé : {ESG_MIN})
        - **Diversification** : elle protège davantage qu'une allocation concentrée — même green.
        - **Argumenter** un choix vaut autant que le calculer : c'est tout le métier de gérant.
        """
    )

    st.caption(
        "Références : Markowitz (1952) · Pedersen, Fitzgibbons & Pomorski (2021) · "
        "Règlement (UE) 2019/2088 (SFDR)"
    )
