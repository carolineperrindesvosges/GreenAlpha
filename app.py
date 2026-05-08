"""
GreenAlpha Challenge — Serious Game M1 Finance, Grenoble IAE
Cours : Investissements et marchés financiers (24h)
"""
import warnings
warnings.filterwarnings('ignore')

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from scipy.optimize import minimize

# ── CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenAlpha Challenge",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PALETTE PASTEL — LILAS + ROSE POUDRÉ ──────────────────────────────────
C_LILAC_DEEP  = '#6B5B95'
C_LILAC       = '#9B89C2'
C_LILAC_MID   = '#B8A8D9'
C_LILAC_SOFT  = '#D5CAE5'
C_LILAC_BG    = '#ECE5F4'
C_LILAC_PALER = '#F6F2FA'

C_ROSE_DEEP   = '#B07A8E'
C_ROSE        = '#D4A0B5'
C_ROSE_MID    = '#E5BFCD'
C_ROSE_SOFT   = '#F0D5DE'
C_ROSE_BG     = '#F8E5EC'
C_ROSE_PALER  = '#FCF5F8'

C_TEXT  = '#2D2A3A'
C_MUTED = '#8A8595'

PASTEL_CMAP = LinearSegmentedColormap.from_list(
    'pastel_esg',
    [(0.0, '#B07A8E'), (0.4, '#E5BFCD'), (0.6, '#C5B8DD'), (1.0, '#6B5B95')],
)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#FBFAFD',
    'axes.grid':        True,
    'grid.alpha':       0.20,
    'grid.color':       '#D5CAE5',
    'font.family':      'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.edgecolor':    '#B8A8D9',
    'axes.labelcolor':   C_TEXT,
    'xtick.color':       C_MUTED,
    'ytick.color':       C_MUTED,
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

# ── LEADERBOARD PARTAGÉ ───────────────────────────────────────────────────
LEADERBOARD_FILE = Path("leaderboard.json")

def load_leaderboard():
    if LEADERBOARD_FILE.exists():
        try:
            return json.loads(LEADERBOARD_FILE.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return []
    return []

def save_to_leaderboard(team_name, score, sharpe_a1, sharpe_a2, esg_a2,
                       sfdr_ok):
    """Met à jour si meilleur score, sinon insère."""
    board = load_leaderboard()
    entry = {
        'team': team_name,
        'score': round(float(score), 1),
        'sharpe_a1': round(float(sharpe_a1), 3),
        'sharpe_a2': round(float(sharpe_a2), 3),
        'esg_a2': round(float(esg_a2), 1),
        'sfdr_ok': bool(sfdr_ok),
        'timestamp': datetime.now().strftime('%H:%M'),
    }
    updated = False
    for i, e in enumerate(board):
        if e['team'].strip().lower() == team_name.strip().lower():
            if entry['score'] > e['score']:
                board[i] = entry
            updated = True
            break
    if not updated:
        board.append(entry)
    board.sort(key=lambda x: -x['score'])
    try:
        LEADERBOARD_FILE.write_text(
            json.dumps(board, indent=2, ensure_ascii=False), encoding='utf-8'
        )
    except OSError:
        pass
    return board

def reset_leaderboard():
    try:
        if LEADERBOARD_FILE.exists():
            LEADERBOARD_FILE.unlink()
    except OSError:
        pass

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
        rets_shock[NAMES.index('TotalEnergies')] -= 0.15
    elif event_label.startswith("2"):
        rets_shock[NAMES.index('Stellantis')] -= 0.20
        esg_shock[NAMES.index('Stellantis')]   = 5
    elif event_label.startswith("3"):
        rets_shock[NAMES.index('Schneider')]   += 0.15
        rets_shock[NAMES.index('Air Liquide')] += 0.10
    return rets_shock, esg_shock

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
        st.session_state.team_name = 'MonÉquipe'
    if 'event_label' not in st.session_state:
        st.session_state.event_label = '1 — Krach sectoriel Énergie'
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    for name in NAMES:
        st.session_state.setdefault(f'a1_{name}', 0)
        st.session_state.setdefault(f'a2_{name}', 0)

init_state()

# ── HEADER ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,{C_LILAC_BG} 0%,{C_LILAC_PALER} 50%,{C_ROSE_BG} 100%);
                padding:28px;border-radius:16px;text-align:center;
                margin-bottom:20px;border:1px solid {C_LILAC_SOFT};
                box-shadow:0 2px 14px rgba(107,91,149,0.08);">
        <h1 style="margin:0;font-weight:600;letter-spacing:0.5px;color:{C_LILAC_DEEP};">
          GreenAlpha Challenge
        </h1>
        <p style="color:{C_ROSE_DEEP};margin:10px 0 0 0;font-size:1.05em;font-style:italic;">
          Une journée chez GreenAlpha Asset Management — place Vendôme, Paris
        </p>
        <p style="color:{C_LILAC};margin:8px 0 0 0;font-size:0.92em;font-style:italic;">
          Garanti sans greenwashing, sans cravate, et avec du vrai café ☕
        </p>
        <p style="color:{C_MUTED};margin:10px 0 0 0;font-size:0.82em;">
          M1 Finance · Grenoble IAE · Cours « Investissements et marchés financiers »
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR (avec leaderboard live) ───────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{C_LILAC_BG},{C_ROSE_BG});
                    padding:14px;border-radius:10px;text-align:center;
                    margin-bottom:14px;border:1px solid {C_LILAC_SOFT};">
            <p style="margin:0;color:{C_LILAC_DEEP};font-weight:600;
                      letter-spacing:1.5px;font-size:0.85em;">
              🏆 CLASSEMENT LIVE
            </p>
            <p style="margin:4px 0 0 0;color:{C_MUTED};font-size:0.72em;
                      font-style:italic;">
              Mise à jour à chaque soumission
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    board = load_leaderboard()

    if not board:
        st.markdown(
            f"""
            <div style="background:{C_LILAC_PALER};padding:14px;
                        border-radius:8px;text-align:center;
                        color:{C_MUTED};font-size:0.85em;font-style:italic;
                        margin-bottom:12px;">
              Aucune équipe n'a encore soumis.<br>
              <span style="font-size:0.8em;">(Les courageux d'abord.)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        medals = ["🥇", "🥈", "🥉"]
        rows_html = ""
        for i, e in enumerate(board[:10]):
            medal = medals[i] if i < 3 else f"<span style='color:{C_MUTED};'>{i+1}.</span>"
            sfdr_tag = (
                f"<span style='color:{C_LILAC_DEEP};font-size:0.7em;'>· Art.8 ✓</span>"
                if e.get('sfdr_ok') else
                f"<span style='color:{C_ROSE_DEEP};font-size:0.7em;'>· Art.8 ✗</span>"
            )
            bg = C_LILAC_BG if i == 0 else (C_ROSE_PALER if i % 2 else C_LILAC_PALER)
            rows_html += f"""
            <div style="background:{bg};padding:8px 10px;border-radius:6px;
                        margin-bottom:4px;font-size:0.85em;">
              <div style="display:flex;justify-content:space-between;">
                <span style="color:{C_TEXT};font-weight:500;">
                  {medal} {e['team'][:18]}
                </span>
                <span style="color:{C_LILAC_DEEP};font-weight:600;">
                  {e['score']:.1f}
                </span>
              </div>
              <div style="color:{C_MUTED};font-size:0.75em;margin-top:2px;">
                Sharpe A2 : {e['sharpe_a2']:.2f} · ESG : {e['esg_a2']:.0f} {sfdr_tag}
              </div>
            </div>
            """
        st.markdown(rows_html, unsafe_allow_html=True)

        if len(board) > 10:
            st.caption(f"… et {len(board)-10} autres équipes plus bas dans le classement.")

    st.button("🔄 Rafraîchir le classement", use_container_width=True,
              help="Au cas où une autre équipe vient de vous doubler.")

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    st.text_input(
        "Nom de votre équipe", key='team_name',
        help="Évitez les noms qui finiront sur LinkedIn. Genre 'AlphaMaximizers69'.",
    )
    st.selectbox(
        "Événement de la journée",
        ['1 — Krach sectoriel Énergie',
         '2 — Scandale ESG Stellantis',
         '3 — Rally Green Tech'],
        key='event_label',
    )
    st.markdown("---")
    st.caption(f"Taux sans risque : **{RF*100:.1f}%**  *(Bund 10 ans, à peu près)*")
    st.caption(f"Seuil ESG SFDR Art. 8 : **{ESG_MIN}**")
    st.caption("Univers : 10 actions Euronext")
    st.markdown("---")
    if st.button("🔄 Réinitialiser tous les poids", use_container_width=True):
        for name in NAMES:
            st.session_state[f'a1_{name}'] = 0
            st.session_state[f'a2_{name}'] = 0
        st.session_state.submitted = False
        st.rerun()

    with st.expander("🛠️ Mode prof (admin)"):
        if st.button("🗑️ Vider le classement", use_container_width=True):
            reset_leaderboard()
            st.success("Classement réinitialisé. Personne n'a rien vu.")
            st.rerun()

rets_shock, esg_shock = get_event_data(st.session_state.event_label)

# ── ONGLETS ───────────────────────────────────────────────────────────────
tab_brief, tab_a1, tab_evt, tab_a2, tab_score, tab_pitch = st.tabs(
    ["☕ 8h30 · Briefing",
     "🎯 11h · Acte 1",
     "📰 12h17 · Breaking news",
     "🌿 15h47 · Acte 2",
     "📊 18h · Score",
     "🎤 18h30 · Comité"]
)

# ─────────────────────────────────────────────────────────────────────────
# BRIEFING — 8h30
# ─────────────────────────────────────────────────────────────────────────
with tab_brief:
    st.markdown(f"""
    <div style="background:{C_LILAC_PALER};border-left:4px solid {C_LILAC};
                padding:18px 20px;border-radius:6px;margin-bottom:20px;">
      <p style="color:{C_MUTED};font-size:0.85em;margin:0 0 6px 0;
                letter-spacing:1px;">📍 LUNDI 8H30 · PLACE VENDÔME, PARIS</p>
      <p style="font-size:1.05em;color:{C_TEXT};margin:0;line-height:1.6;">
        Vous poussez la lourde porte de <b>GreenAlpha Asset Management</b>.
        Boutique d'investissement créée il y a trois ans par d'anciens gérants
        de chez Carmignac qui en avaient marre des cravates. Vingt collaborateurs,
        1,2 milliard d'encours, et une réputation : <i>du green qui ne triche pas</i>
        (la barre est étonnamment basse dans le métier).
      </p>
      <p style="font-size:1.05em;color:{C_TEXT};margin:14px 0 0 0;line-height:1.6;">
        <b>Élise Marchand</b>, votre directrice de la gestion, vous tend un café
        tiède (la machine est en panne depuis vendredi, c'est l'IT qui s'en occupe,
        donc autant dire jamais) et s'assoit en face de vous.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{C_ROSE_PALER};border-left:4px solid {C_ROSE};
                padding:16px 20px;border-radius:6px;margin-bottom:20px;
                font-style:italic;color:{C_TEXT};line-height:1.65;">
      « Bienvenue dans l'équipe.<br><br>
      On vient de boucler une levée de <b>200 millions d'euros</b> auprès de
      family offices européens — vous savez, ces gens qui font tourner leur
      patrimoine entre Genève et Monaco et qui veulent <i>aussi</i> sauver la
      planète. Le mandat est clair : un fonds <b>SFDR Article 8</b>,
      surperformance attendue, et zéro tolérance au greenwashing.<br><br>
      Les investisseurs nous regardent, la presse aussi, et l'AMF — disons —
      garde un œil bienveillant. Je vous laisse la matinée pour proposer une
      <b>première allocation</b>. On en parle à 11h.<br><br>
      Bon courage. Et fermez la porte en sortant, le radiateur fuit. »
      <p style="text-align:right;margin:10px 0 0 0;font-style:normal;
                color:{C_MUTED};font-size:0.85em;">— Élise Marchand,
                Directrice de la gestion</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Votre univers d'investissement")
    st.caption("Dix actions européennes sélectionnées par le comité ESG la semaine "
               "dernière, autour d'un déjeuner qui a duré trois heures.")

    df_assets = pd.DataFrame({
        'Actif': NAMES,
        'Secteur': [ASSETS[a]['sector'] for a in NAMES],
        'Rendement (%)':  (RETS * 100).round(1),
        'Volatilité (%)': (VOLS * 100).round(1),
        'Score ESG /100': ESG,
    })

    def color_esg(val):
        if val >= 65: return f'background-color:{C_LILAC_BG};color:{C_LILAC_DEEP};font-weight:600'
        if val >= 45: return f'background-color:{C_ROSE_PALER};color:{C_ROSE_DEEP}'
        return f'background-color:{C_ROSE_SOFT};color:{C_ROSE_DEEP};font-weight:600'

    st.dataframe(
        df_assets.style.map(color_esg, subset=['Score ESG /100']),
        use_container_width=True, hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sc = ax.scatter(VOLS * 100, RETS * 100, c=ESG, cmap=PASTEL_CMAP,
                    vmin=20, vmax=90, s=200, zorder=5,
                    edgecolors='white', lw=2)
    cbar = plt.colorbar(sc, ax=ax, label='Score ESG')
    cbar.ax.tick_params(labelsize=8)
    for i, name in enumerate(NAMES):
        ax.annotate(name, (VOLS[i]*100, RETS[i]*100),
                    textcoords='offset points', xytext=(8, 5),
                    fontsize=9, color=C_TEXT)
    ax.axhline(np.mean(RETS)*100, color=C_LILAC_MID, ls=':', alpha=0.6,
               label='Rdt. moyen')
    ax.axvline(np.mean(VOLS)*100, color=C_LILAC_MID, ls='--', alpha=0.6,
               label='Vol. moyenne')
    ax.set_xlabel('Volatilité (%)'); ax.set_ylabel('Rendement espéré (%)')
    ax.set_title("Espace risque / rendement — couleur = score ESG",
                 fontweight='bold', color=C_LILAC_DEEP)
    ax.legend(fontsize=9); fig.tight_layout()
    st.pyplot(fig)

    with st.expander("💡 Conseils d'Élise avant de commencer (vraiment, lisez)"):
        st.markdown(f"""
        - Regardez les **vertes peu rentables** (L'Oréal, Schneider, Air Liquide)
          face aux **brunes mais lucratives** (Stellantis, TotalEnergies) :
          c'est l'arbitrage central de votre métier. Et accessoirement, le
          sujet de 80% des entretiens en gestion d'actifs.
        - Une **volatilité élevée** ne veut pas dire mauvais actif —
          la **diversification** peut le rendre indispensable. Markowitz l'a
          démontré en 1952. Ça lui a pris 38 ans pour avoir le Nobel. Soyez patients.
        - Le **score ESG du portefeuille** est la moyenne *pondérée* :
          un peu de Stellantis tire la moyenne vers le bas très vite. Un peu
          comme un seul stagiaire en short au comité d'investissement.
        """)

# ─────────────────────────────────────────────────────────────────────────
# ACTE 1 — 11h
# ─────────────────────────────────────────────────────────────────────────
with tab_a1:
    st.markdown(f"""
    <div style="background:{C_LILAC_PALER};border-left:4px solid {C_LILAC};
                padding:18px 20px;border-radius:6px;margin-bottom:20px;">
      <p style="color:{C_MUTED};font-size:0.85em;margin:0 0 6px 0;
                letter-spacing:1px;">📍 11H00 · OPEN-SPACE GREENALPHA</p>
      <p style="color:{C_TEXT};margin:0;line-height:1.6;">
        Élise est en réunion avec les avocats du fonds. Trois avocats. Pour vingt
        collaborateurs. C'est ça, la finance moderne. Vous avez carte blanche
        jusqu'à midi, et personne ne va venir vous embêter — sauf le stagiaire
        qui cherche le code du Wi-Fi pour la quatrième fois.
      </p>
      <p style="color:{C_TEXT};margin:12px 0 0 0;line-height:1.6;
                font-style:italic;">
        Mémo qu'elle vous a laissé sur le bureau : <i>« Pour cette première
        allocation, oubliez les contraintes ESG officielles —
        <b>maximisez le ratio de Sharpe</b>. Mais gardez en tête : nous sommes
        un fonds Article 8, l'œil de l'AMF est sur nous. Et l'AMF, contrairement
        aux investisseurs, lit les rapports en entier. »</i>
      </p>
    </div>
    """, unsafe_allow_html=True)

    fv_ref, fr_ref = frontier()
    w_opt_ref = optimize()
    r_ref, v_ref = port_stats(w_opt_ref)
    sharpe_ref = (r_ref - RF) / v_ref

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(VOLS*100, RETS*100, c=ESG, cmap=PASTEL_CMAP,
               vmin=20, vmax=90, s=130, zorder=5,
               edgecolors='white', lw=1.5)
    for i, name in enumerate(NAMES):
        ax.annotate(name, (VOLS[i]*100, RETS[i]*100),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8.5, color=C_TEXT)
    ax.plot(fv_ref*100, fr_ref*100, '--', color=C_LILAC_DEEP, lw=2.4,
            label='Frontière efficiente')
    ax.scatter([v_ref*100], [r_ref*100], marker='*', s=380,
               color=C_ROSE_DEEP, zorder=6,
               edgecolor='white', lw=1.5,
               label=f"Max Sharpe = {sharpe_ref:.3f}")
    x_cml = np.linspace(0, max(VOLS)*100*1.1, 100)
    ax.plot(x_cml, RF*100 + sharpe_ref*x_cml, ':',
            color=C_ROSE, alpha=0.7, label='CML')
    ax.set_xlabel('Volatilité (%)'); ax.set_ylabel('Rendement espéré (%)')
    ax.set_title('Frontière efficiente — Acte 1',
                 fontweight='bold', color=C_LILAC_DEEP)
    ax.legend(fontsize=9); fig.tight_layout()
    st.pyplot(fig)

    st.info(
        f"💡 **Optimum théorique** (l'œil du quant senior, qui a pris trois cafés) : "
        f"Rdt={r_ref*100:.2f}% · Vol={v_ref*100:.2f}% · "
        f"Sharpe={sharpe_ref:.3f} · ESG={w_opt_ref@ESG:.1f}"
    )

    bcol1, bcol2, bcol3 = st.columns(3)
    if bcol1.button("⚖️ Équipondération (10% chacun)", use_container_width=True,
                    help="La solution lâche. Ne fait jamais perdre, ne fait jamais gagner."):
        set_weights_from_array('a1', np.ones(N) / N); st.rerun()
    if bcol2.button("🎯 Copier l'optimum théorique", use_container_width=True,
                    help="Personne ne saura. Mais Élise verra."):
        set_weights_from_array('a1', w_opt_ref); st.rerun()
    if bcol3.button("🧹 Effacer Acte 1", use_container_width=True):
        for name in NAMES: st.session_state[f'a1_{name}'] = 0
        st.rerun()

    st.markdown("#### ✏️ Votre allocation (en %)")
    cols = st.columns(2)
    for i, name in enumerate(NAMES):
        with cols[i % 2]:
            st.number_input(name, min_value=0, max_value=100, step=5, key=f'a1_{name}')

    w1, total1 = weights_from_state('a1')
    st.progress(min(total1 / 100, 1.0), text=f"Somme des poids : {total1:.0f}% / 100%")

    if abs(total1 - 100) > 0.1:
        st.error("⚠️ La somme des poids doit être exactement égale à 100%. "
                 "On ne fait pas de short, on n'emprunte pas, on a l'argent ou on l'a pas.")
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

# ─────────────────────────────────────────────────────────────────────────
# ÉVÉNEMENT — 12h17
# ─────────────────────────────────────────────────────────────────────────
with tab_evt:
    label = st.session_state.event_label

    if label.startswith("1"):
        narrative_html = f"""
        <div style="background:{C_ROSE_PALER};border:2px solid {C_ROSE_DEEP};
                    padding:20px;border-radius:8px;margin-bottom:20px;
                    box-shadow:0 2px 8px rgba(176,122,142,0.15);">
          <p style="color:{C_ROSE_DEEP};font-weight:bold;font-size:0.85em;
                    letter-spacing:2px;margin:0 0 10px 0;">
            🚨 12H17 · BREAKING NEWS · BLOOMBERG
          </p>
          <h3 style="color:{C_LILAC_DEEP};margin:0 0 12px 0;">
            Le gouvernement annonce une taxe surprise sur les superprofits pétroliers
          </h3>
          <p style="color:{C_TEXT};line-height:1.6;margin:0 0 12px 0;">
            L'écran Bloomberg passe au rouge. X (anciennement Twitter, anciennement
            sérieux) s'enflamme. Le CAC plonge à l'ouverture des États-Unis. Quelque
            part, un éditorialiste rédige déjà un papier intitulé <i>« Fallait s'y
            attendre »</i>.
          </p>
          <p style="color:{C_TEXT};line-height:1.6;margin:0 0 12px 0;
                    font-size:1.1em;">
            <b>TotalEnergies : −15%</b> en deux heures. Le PDG est <i>« en réunion »</i>.
          </p>
          <p style="color:{C_TEXT};font-style:italic;line-height:1.6;
                    margin:0;border-left:3px solid {C_LILAC};padding-left:12px;">
            Votre téléphone vibre. C'est Élise.<br>
            « Ne paniquez pas. Mais regardez votre exposition Énergie.
            On en parle dans 30 minutes. Et non, ce n'est pas le moment d'aller
            chercher un sandwich. »
          </p>
        </div>
        """
    elif label.startswith("2"):
        narrative_html = f"""
        <div style="background:{C_ROSE_PALER};border:2px solid {C_ROSE_DEEP};
                    padding:20px;border-radius:8px;margin-bottom:20px;
                    box-shadow:0 2px 8px rgba(176,122,142,0.15);">
          <p style="color:{C_ROSE_DEEP};font-weight:bold;font-size:0.85em;
                    letter-spacing:2px;margin:0 0 10px 0;">
            💥 14H03 · SCANDALE · LE MONDE / FT
          </p>
          <h3 style="color:{C_LILAC_DEEP};margin:0 0 12px 0;">
            Stellantis a falsifié ses tests d'émissions CO₂ depuis 2022
          </h3>
          <p style="color:{C_TEXT};line-height:1.6;margin:0 0 12px 0;">
            Une enquête conjointe <i>Le Monde</i> / <i>Financial Times</i> sort à
            14h pile (les journalistes savent ce qu'ils font). Documents internes,
            témoignages d'ingénieurs, mails de la direction qui commencent par
            <i>« Surtout, ne mettez pas ça par écrit »</i> — mis par écrit.
            La Commission européenne convoque la direction pour demain. Des historiens
            de l'industrie automobile parlent déjà de <i>« Dieselgate 2 : l'électrique
            contre-attaque »</i>.
          </p>
          <p style="color:{C_TEXT};line-height:1.6;margin:0 0 12px 0;
                    font-size:1.1em;">
            <b>Stellantis : −20%</b>. Score ESG dégradé de
            <b>29 → 5</b> par les agences de notation dans la foulée. Cinq.
            Sur cent.
          </p>
          <p style="color:{C_TEXT};font-style:italic;line-height:1.6;
                    margin:0;border-left:3px solid {C_LILAC};padding-left:12px;">
            Élise débarque dans votre bureau, livide.<br>
            « Si on a Stellantis dans le portefeuille, on est mort sur l'ESG.
            Réagissez. Et fermez Bloomberg, ça stresse tout le monde. »
          </p>
        </div>
        """
    else:
        narrative_html = f"""
        <div style="background:{C_LILAC_PALER};border:2px solid {C_LILAC_DEEP};
                    padding:20px;border-radius:8px;margin-bottom:20px;
                    box-shadow:0 2px 8px rgba(107,91,149,0.15);">
          <p style="color:{C_LILAC_DEEP};font-weight:bold;font-size:0.85em;
                    letter-spacing:2px;margin:0 0 10px 0;">
            🌿 15H22 · FLASH · COMMISSION EUROPÉENNE
          </p>
          <h3 style="color:{C_LILAC_DEEP};margin:0 0 12px 0;">
            Plan de 800 milliards d'euros pour la transition écologique
          </h3>
          <p style="color:{C_TEXT};line-height:1.6;margin:0 0 12px 0;">
            Ursula von der Leyen dévoile en conférence de presse un plan massif
            d'investissement vert. Les analystes sortent leurs notes en quinze
            minutes (Goldman avait la sienne prête depuis trois semaines, mais
            chut). Les valeurs green s'envolent. Quelque part, un fonds qui a
            shorté Schneider hier soir <i>regrette ses choix</i>.
          </p>
          <p style="color:{C_TEXT};line-height:1.6;margin:0 0 12px 0;
                    font-size:1.1em;">
            <b>Schneider Electric : +15%</b> · <b>Air Liquide : +10%</b>.
          </p>
          <p style="color:{C_TEXT};font-style:italic;line-height:1.6;
                    margin:0;border-left:3px solid {C_ROSE};padding-left:12px;">
            Ping. Slack. Élise.<br>
            « On profite ou on a raté le train ? Vos chiffres dans 5 minutes.
            Et ce n'est pas une question rhétorique. »
          </p>
        </div>
        """

    st.markdown(narrative_html, unsafe_allow_html=True)

    w1, total1 = weights_from_state('a1')
    if abs(total1 - 100) > 0.1:
        st.info("⏳ Complétez d'abord votre Acte 1 pour mesurer l'impact. "
                "Sinon on ne saura pas combien vous avez perdu, et c'est triste.")
    else:
        r_av, v_av = port_stats(w1, RETS)
        r_ap, v_ap = port_stats(w1, rets_shock)
        sh_av = (r_av - RF) / v_av
        sh_ap = (r_ap - RF) / v_ap

        st.markdown("### 💼 Impact sur votre portefeuille Acte 1")

        c1, c2 = st.columns(2)
        c1.metric("Rendement avant choc", f"{r_av*100:.2f}%")
        c2.metric("Rendement après choc", f"{r_ap*100:.2f}%",
                  delta=f"{(r_ap - r_av)*100:+.2f} pp")
        c3, c4 = st.columns(2)
        c3.metric("Sharpe avant choc", f"{sh_av:.3f}")
        c4.metric("Sharpe après choc", f"{sh_ap:.3f}",
                  delta=f"{sh_ap - sh_av:+.3f}")

        st.markdown("#### Détail actif par actif")
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

# ─────────────────────────────────────────────────────────────────────────
# ACTE 2 — 15h47
# ─────────────────────────────────────────────────────────────────────────
with tab_a2:
    st.markdown(f"""
    <div style="background:{C_LILAC_PALER};border-left:4px solid {C_LILAC_DEEP};
                padding:18px 20px;border-radius:6px;margin-bottom:20px;">
      <p style="color:{C_MUTED};font-size:0.85em;margin:0 0 6px 0;
                letter-spacing:1px;">📍 15H47 · SALLE DE RÉUNION B12</p>
      <p style="color:{C_TEXT};margin:0;line-height:1.6;">
        Réunion d'urgence du comité d'investissement. Six personnes autour de la
        table. Le risk manager triture nerveusement son stylo — le risk manager
        déteste les surprises, c'est pour ça qu'on l'appelle le risk manager.
        Élise pose ses lunettes sur le dossier ouvert devant elle.
      </p>
      <p style="color:{C_TEXT};font-style:italic;line-height:1.65;
                margin:14px 0 0 0;border-left:3px solid {C_ROSE};
                padding-left:14px;">
        « Décision prise. On verrouille la classification <b>SFDR Article 8</b>
        — score ESG moyen ≥ <b>{ESG_MIN}</b>, contrainte officielle. Pas une
        recommandation. Une contrainte.<br><br>
        À vous de recomposer le portefeuille. Je veux votre nouvelle allocation
        à <b>17h</b>. Le rendement ne suffit plus. Et oui, je sais qu'il est
        15h47. Bienvenue dans la finance. »
      </p>
    </div>
    """, unsafe_allow_html=True)

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
    ax.scatter(VOLS*100, rets_shock*100, c=esg_shock, cmap=PASTEL_CMAP,
               vmin=20, vmax=90, s=130, zorder=5, edgecolors='white', lw=1.5)
    for i, name in enumerate(NAMES):
        ax.annotate(name, (VOLS[i]*100, rets_shock[i]*100),
                    textcoords='offset points', xytext=(6, 3),
                    fontsize=8.5, color=C_TEXT)
    ax.plot(fv_libre*100, fr_libre*100, '--', color=C_ROSE_DEEP, lw=1.8,
            label='Frontière libre')
    ax.plot(fv_esg*100,   fr_esg*100,   '-',  color=C_LILAC_DEEP, lw=2.4,
            label=f'Frontière ESG ≥ {ESG_MIN}')
    ax.scatter([v_libre*100], [r_libre*100], marker='*', s=300,
               color=C_ROSE_DEEP, zorder=6, edgecolor='white', lw=1.2,
               label=f'Max Sharpe libre ({sh_libre:.3f})')
    ax.scatter([v_esg*100],   [r_esg*100],   marker='*', s=300,
               color=C_LILAC_DEEP, zorder=6, edgecolor='white', lw=1.2,
               label=f'Max Sharpe ESG ({sh_esg:.3f})')
    ax.set_xlabel('Volatilité (%)'); ax.set_ylabel('Rendement (%)')
    ax.set_title('Frontières après l\'événement',
                 fontweight='bold', color=C_LILAC_DEEP)
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    cats = ['Sharpe', 'Rdt (%)', 'Vol (%)', 'ESG']
    v_l = [sh_libre, r_libre*100, v_libre*100, w_opt_libre @ esg_shock]
    v_e = [sh_esg,   r_esg*100,   v_esg*100,   w_opt_esg   @ esg_shock]
    x = np.arange(len(cats)); wb = 0.36
    b1 = ax2.bar(x - wb/2, v_l, wb, label='Libre',           color=C_ROSE_DEEP, alpha=0.85)
    b2 = ax2.bar(x + wb/2, v_e, wb, label=f'ESG ≥ {ESG_MIN}', color=C_LILAC_DEEP, alpha=0.85)
    for b in list(b1) + list(b2):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15,
                 f'{b.get_height():.2f}', ha='center', va='bottom',
                 fontsize=8.5, color=C_TEXT)
    ax2.set_xticks(x); ax2.set_xticklabels(cats); ax2.legend()
    ax2.set_ylim(0, max(max(v_l), max(v_e)) * 1.22)
    ax2.set_title(f'Comparaison\nCoût ESG sur Sharpe : {cout_esg_pct:.1f}%',
                  fontweight='bold', color=C_LILAC_DEEP)
    fig.tight_layout()
    st.pyplot(fig)

    st.info(f"💡 **Le coût de la conscience verte** : la contrainte ESG fait perdre "
            f"**{cout_esg_pct:.1f}%** de ratio de Sharpe. Élise va vouloir une "
            f"explication. Préparez-vous.")

    bcol1, bcol2, bcol3 = st.columns(3)
    if bcol1.button("⚖️ Équipondération", key='a2_eq', use_container_width=True):
        set_weights_from_array('a2', np.ones(N) / N); st.rerun()
    if bcol2.button("🎯 Copier l'optimum ESG", key='a2_opt', use_container_width=True,
                    help="Vous savez que c'est de la triche. Élise aussi."):
        set_weights_from_array('a2', w_opt_esg); st.rerun()
    if bcol3.button("🧹 Effacer Acte 2", key='a2_clear', use_container_width=True):
        for name in NAMES: st.session_state[f'a2_{name}'] = 0
        st.rerun()

    st.markdown("#### ✏️ Nouvelle allocation — Acte 2 (en %)")
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
            st.error(f"❌ **Contrainte SFDR violée** : ESG = {esg2_val:.1f} < {ESG_MIN}. "
                     f"Le label Article 8 est retiré au fonds. Le service comm' rédige "
                     f"déjà le communiqué de crise.")
        else:
            st.success(f"✅ **Contrainte SFDR respectée** : ESG = {esg2_val:.1f} ≥ {ESG_MIN}. "
                       f"Le label est sauvé. Élise vous offrira peut-être un café.")

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

# ─────────────────────────────────────────────────────────────────────────
# SCORE — 18h
# ─────────────────────────────────────────────────────────────────────────
with tab_score:
    st.markdown(f"""
    <div style="background:{C_LILAC_PALER};border-left:4px solid {C_LILAC};
                padding:18px 20px;border-radius:6px;margin-bottom:20px;">
      <p style="color:{C_MUTED};font-size:0.85em;margin:0 0 6px 0;
                letter-spacing:1px;">📍 18H00 · DEBRIEF DE FIN DE JOURNÉE</p>
      <p style="color:{C_TEXT};margin:0;line-height:1.6;">
        Le marché ferme. Élise compile votre performance pour le comité.
        Le DG est déjà parti — il avait un dîner. Mais Élise reste. Élise reste
        toujours. C'est l'heure du verdict.
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🧮 Comment est calculé le score ? (lisez avant de râler)"):
        st.markdown(f"""
        Pour **chaque acte** (sur 80 points) :
        - **Sharpe** : `min(Sharpe / 1.5, 1) × 50` → max **50 pts**
        - **ESG**    : `min(ESG / 100, 1) × 30` → max **30 pts**

        Pour **l'Acte 2 uniquement** :
        - 🎁 Bonus de **+10 pts** si ESG ≥ 65 (au-delà de la contrainte — vous êtes meilleurs que la loi vous demande, bravo)
        - ❌ Pénalité de **−15 pts** si ESG < {ESG_MIN} (label retiré, le DG vous regarde de travers)

        **Score total** : Acte 1 + Acte 2, sur **160 points**.
        """)

    w1, total1 = weights_from_state('a1')
    w2, total2 = weights_from_state('a2')
    esg2_val = float(w2 @ esg_shock)

    if abs(total1 - 100) > 0.1 or abs(total2 - 100) > 0.1:
        st.info("⏳ Complétez Acte 1 et Acte 2 (somme = 100%) pour afficher le score. "
                "Le suspense est insoutenable, on sait.")
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
        if score_total >= 130:
            verdict = (f"🌟 **Élise sourit.** « Pas mal pour un premier jour. On en "
                       f"reparle au comité de carrière. » *(Traduction RH : "
                       f"on vous garde.)*")
        elif score_total >= 100:
            verdict = (f"👍 **Élise hoche la tête.** « Vous avez compris l'arbitrage. "
                       f"Vous êtes invités au pot du vendredi. » *(C'est un compliment.)*")
        elif score_total >= 70:
            verdict = (f"🤔 **Élise vous regarde un peu trop longtemps.** « Il y a "
                       f"des leçons à tirer. On débrief lundi. » *(Préparez-vous "
                       f"un argumentaire.)*")
        elif score_total >= 40:
            verdict = (f"😬 **Silence dans le bureau d'Élise.** « On va revoir les "
                       f"fondamentaux ensemble. Tous les deux. » *(Mauvais signe.)*")
        else:
            verdict = (f"☠️ **Le DG se sert un whisky. À 18h. Lundi.** Les avocats "
                       f"sont déjà prévenus. *(Très mauvais signe.)*")

        st.success(f"🏆 **{st.session_state.team_name} : {score_total:.1f} / 160**")
        st.markdown(f"<p style='font-size:1.05em;color:{C_TEXT};margin-top:10px;'>{verdict}</p>",
                    unsafe_allow_html=True)
        st.progress(min(max(score_total / 160, 0.0), 1.0))

        st.markdown("---")
        st.markdown("### 📤 Soumettre votre score au classement")

        col_sub1, col_sub2 = st.columns([2, 1])
        with col_sub1:
            st.caption(
                f"Vous allez soumettre sous le nom : **{st.session_state.team_name}**. "
                "Si une équipe porte déjà ce nom, seul le meilleur score est conservé. "
                "Vous pouvez resoumettre plusieurs fois — on ne juge pas (Élise oui)."
            )
        with col_sub2:
            if st.button("🚀 Soumettre au classement", use_container_width=True,
                         type='primary'):
                save_to_leaderboard(
                    team_name=st.session_state.team_name,
                    score=score_total,
                    sharpe_a1=s1['sharpe'],
                    sharpe_a2=s2['sharpe'],
                    esg_a2=esg2_val,
                    sfdr_ok=(esg2_val >= ESG_MIN),
                )
                st.session_state.submitted = True
                st.rerun()

        if st.session_state.submitted:
            st.success(
                "✅ Score soumis. Le classement à gauche est à jour. "
                "Maintenant priez pour qu'aucune autre équipe ne fasse mieux."
            )

        # Mini-récap du classement dans cet onglet
        board = load_leaderboard()
        if board:
            st.markdown("#### 🏆 Top 5 actuel")
            df_top = pd.DataFrame(board[:5])[
                ['team', 'score', 'sharpe_a2', 'esg_a2', 'sfdr_ok', 'timestamp']
            ]
            df_top.columns = ['Équipe', 'Score', 'Sharpe A2', 'ESG A2', 'Art.8 ✓', 'Heure']
            df_top.insert(0, 'Rang', range(1, len(df_top) + 1))
            st.dataframe(df_top, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────
# PITCH — 18h30
# ─────────────────────────────────────────────────────────────────────────
with tab_pitch:
    st.markdown(f"""
    <div style="background:{C_ROSE_PALER};border-left:4px solid {C_ROSE_DEEP};
                padding:18px 20px;border-radius:6px;margin-bottom:20px;">
      <p style="color:{C_MUTED};font-size:0.85em;margin:0 0 6px 0;
                letter-spacing:1px;">📍 18H30 · COMITÉ D'INVESTISSEMENT</p>
      <p style="color:{C_TEXT};margin:0;line-height:1.6;">
        Vous entrez dans la salle du comité. Autour de la table : Élise, le DG
        (qui a fini par rester), deux avocats SFDR (qui ne disent rien mais
        prennent des notes), le risk manager (toujours son stylo), et le responsable
        de la relation investisseurs (sourire de circonstance). Tous attendent.
      </p>
      <p style="color:{C_TEXT};margin:14px 0 0 0;line-height:1.6;
                font-style:italic;">
        Vous avez <b>10 minutes</b> pour défendre votre allocation. Pas de slides
        — Élise hait PowerPoint depuis qu'un stagiaire a animé une transition en
        2019. Juste vos chiffres et votre raisonnement.
      </p>
    </div>
    """, unsafe_allow_html=True)

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
            cols = PASTEL_CMAP(np.array(esgs) / 100) if len(esgs) else []
            bars = ax.barh(lbls, vals, color=cols, edgecolor='white', linewidth=1.8)
            for bar, e in zip(bars, esgs):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        f'ESG {e}', va='center', fontsize=8.5, color=C_TEXT)
            r, v = port_stats(w, rets_used)
            sh = (r - RF) / v if v > 1e-8 else 0
            ax.set_xlabel('Poids (%)')
            ax.set_title(f'{titre}\nSharpe={sh:.3f} · ESG={w@esg_used:.1f}',
                         fontweight='bold', color=C_LILAC_DEEP)
            ax.set_xlim(0, max(vals)*1.25 if vals else 10)
        fig.suptitle('Évolution Acte 1 → Acte 2',
                     fontsize=12, fontweight='bold', color=C_LILAC_DEEP, y=1.02)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown(f"""
    <div style="background:{C_LILAC_PALER};padding:16px 20px;
                border-radius:6px;margin-top:10px;">
      <h4 style="color:{C_LILAC_DEEP};margin:0 0 12px 0;">
        🎤 Les questions du comité (préparez-vous, elles tombent toutes)
      </h4>
      <ol style="color:{C_TEXT};line-height:1.8;margin:0;padding-left:20px;">
        <li>Quels actifs avez-vous <b>renforcés ou réduits</b> entre l'Acte 1
            et l'Acte 2 ? Pourquoi ? <i>(« Pour faire joli » n'est pas une réponse.)</i></li>
        <li>Quel est votre <b>coût ESG</b> en perte de Sharpe ? Est-ce justifiable
            auprès des investisseurs ? <i>(Astuce : oui, si vous savez l'expliquer.)</i></li>
        <li>Si la Commission durcit le seuil ESG <b>à 70</b> demain, que feriez-vous ?
            <i>(Spoiler : c'est jamais « rien ».)</i></li>
        <li>Comment votre allocation se compare-t-elle à un fonds <b>équipondéré</b> ?
            <i>(Si la réponse est « moins bien », il faut le dire.)</i></li>
        <li><i>Question piège du DG :</i> pourquoi ne pas sortir entièrement
            des actifs à faible ESG ? <i>(Réfléchissez avant de répondre. Vraiment.)</i></li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Le verdict du quant senior — portefeuille optimal sous contrainte")

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
        #### 💡 Ce qu'il faut retenir de la journée

        - Le **coût ESG** sur le Sharpe ressort à **{cout:.1f}%** : c'est le prix
          de la conformité Article 8. Ce n'est ni rien, ni rédhibitoire. C'est
          votre métier de l'expliquer.
        - L'ESG moyen optimal sous contrainte : **{w_opt_esg @ esg_shock:.1f}**
          (seuil imposé : {ESG_MIN}). On ne fait pas plus que nécessaire — c'est
          ça, l'optimisation. Le moralisme, c'est gratuit ; l'arbitrage,
          c'est rémunéré.
        - **La diversification protège** plus qu'une allocation concentrée — même
          verte. Une seule action peut couler un fonds (cf. Stellantis, et tous
          les autres avant lui : Wirecard, Enron, Lehman… la liste est longue
          et elle s'allongera).
        - **Argumenter** un choix d'allocation vaut autant que le calculer. C'est
          tout le métier. Le code optimise. Vous, vous convainquez.
        """
    )

    st.caption(
        "Références académiques (les vraies) : Markowitz (1952), *Portfolio Selection*, "
        "*Journal of Finance* · Sharpe (1964), *Capital Asset Prices*, *Journal of Finance* · "
        "Pedersen, Fitzgibbons & Pomorski (2021), *Responsible investing: The ESG-efficient "
        "frontier*, *Journal of Financial Economics* · Règlement (UE) 2019/2088 (SFDR), Art. 8."
    )
