"""Shared design system for the GreenLeaf dashboard — the single source of visual truth.

Every tab imports from here and is built ONLY from these primitives; no tab defines its
own colours, CSS, or chart styling. This module holds NO data and NO numbers.

Aesthetic: refined "Lululemon editorial" — cream background, deep-forest + sage accents,
Georgia serif headlines, generous whitespace, hairline dividers, tiny uppercase
letter-spaced labels. Calm, premium, trustworthy. Not bold/portfolio, not neon, not dark.

API:
    inject_css()                              one global CSS block (call once per tab)
    brand_bar(eyebrow)                        leaf mark + GREENLEAF wordmark + section eyebrow
    section_header(label, title, sub=None)    eyebrow + serif title + hairline
    kpi_row([kpi(...), ...])                  row of editorial KPI tiles
    kpi(value, label, sub=None, accent=SAGE)  -> dict consumed by kpi_row
    card_title(text)                          tiny uppercase label above a chart
    caveat_note(text)                         sage left-rule caveat callout (use for every 🟡)
    note(text)                                muted caption line
    hairline()                                a hairline divider
    style_fig(fig, height, ...)               apply the chart theme to EVERY Plotly figure
    CAT / SEQ / SEQ_WARM / colour tokens      palette
"""
import html as _html

import streamlit as st

# --- Colour tokens ------------------------------------------------------------
INK = "#1F2A1F"      # deep forest text
SAGE = "#3D5A3D"     # primary accent
CREAM = "#F5F2EC"    # page background
PAPER = "#FBFAF5"    # card surface (a hair lighter than cream)
TAUPE = "#5A5A55"    # secondary text
MUTED = "#88897F"    # tertiary text / captions
HAIR = "#D9D4C7"     # hairline rules / borders
TERRA = "#8A5A3D"    # warm secondary accent
INDIGO = "#5A5A8A"   # cool secondary accent
SAGE_SOFT = "#6E8B6E"

# Plotly colourways (forest/earth — never default Plotly blue, never neon)
CAT = [SAGE, SAGE_SOFT, "#A7B79E", TERRA, INDIGO, TAUPE, "#B08968"]
SEQ = [[0.0, "#EFE9DC"], [1.0, SAGE]]
SEQ_WARM = [[0.0, "#EFE9DC"], [1.0, "#C39072"]]

SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif"

# --- One global CSS block -----------------------------------------------------
_CSS = f"""
<style>
:root {{ --ink:{INK}; --sage:{SAGE}; --cream:{CREAM}; --paper:{PAPER};
  --taupe:{TAUPE}; --muted:{MUTED}; --hair:{HAIR}; --terra:{TERRA}; --indigo:{INDIGO}; }}

.stApp {{ background-color:{CREAM} !important; }}
[data-testid="stHeader"] {{ background:transparent !important; height:0 !important; }}
[data-testid="stToolbar"] {{ right:8px; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ max-width:1240px !important; padding:1.6rem 2.4rem 4rem !important; }}
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {{ background:{CREAM} !important; }}

/* Typography */
.stApp, .stApp p, .stApp li, [data-testid="stMarkdownContainer"] {{ color:{INK}; }}
.gl-eyebrow {{ text-transform:uppercase; letter-spacing:2.6px; font-size:0.68rem;
  color:{TAUPE}; margin:0; font-family:{SANS}; }}
.gl-h1 {{ font-family:{SERIF}; font-weight:400; font-size:34px; line-height:1.1;
  color:{INK}; margin:.15rem 0 0; }}
.gl-h2 {{ font-family:{SERIF}; font-weight:400; font-size:25px; line-height:1.12;
  color:{INK}; margin:.1rem 0 0; }}
.gl-sub {{ font-family:{SERIF}; font-weight:500; font-size:16px; color:{TAUPE};
  margin:.4rem 0 0; line-height:1.4; }}
.gl-hair {{ border:none; border-top:1px solid {HAIR}; margin:14px 0 20px; }}
.gl-cardtitle {{ text-transform:uppercase; letter-spacing:1.8px; font-size:0.66rem;
  color:{TAUPE}; margin:6px 0 2px; font-family:{SANS}; }}
.gl-note {{ font-size:0.81rem; color:{MUTED}; line-height:1.55; margin:.3rem 0 .2rem;
  font-family:{SANS}; }}

/* Brand bar */
.gl-brand {{ display:flex; align-items:center; gap:11px; margin-bottom:2px; }}
.gl-leaf {{ width:28px; height:28px; border-radius:50%; background:{SAGE}; color:{CREAM};
  display:flex; align-items:center; justify-content:center; font-size:14px; flex:0 0 auto; }}
.gl-word {{ font-family:{SERIF}; font-size:1.05rem; letter-spacing:.5px; color:{INK}; }}
.gl-brand .gl-eyebrow {{ margin-left:auto; }}

/* KPI tiles */
.gl-kpis {{ display:flex; gap:14px; margin:6px 0 4px; }}
.gl-kpi {{ flex:1; background:{PAPER}; border:1px solid {HAIR}; border-top:2px solid {SAGE};
  padding:15px 18px 14px; }}
.gl-kpi-label {{ text-transform:uppercase; letter-spacing:1.4px; font-size:0.6rem;
  color:{TAUPE}; font-family:{SANS}; }}
.gl-kpi-value {{ font-family:{SERIF}; font-size:1.62rem; color:{INK}; line-height:1.15;
  margin:.28rem 0 .1rem; }}
.gl-kpi-sub {{ font-size:0.74rem; color:{MUTED}; font-family:{SANS}; }}

/* Caveat callout */
.gl-caveat {{ border-left:2px solid {SAGE}; background:{PAPER}; padding:10px 16px;
  margin:.3rem 0 .2rem; }}
.gl-caveat .gl-cardtitle {{ color:{SAGE}; margin-top:0; }}
.gl-caveat p {{ color:{INK}; margin:.2rem 0 0; font-size:0.86rem; line-height:1.55;
  font-family:{SANS}; }}

/* Plotly chart container -> hairline card */
[data-testid="stPlotlyChart"] {{ border:1px solid {HAIR}; background:{PAPER};
  padding:6px 10px 2px; }}

/* Native widgets: editorial, hairline, sage focus */
[data-testid="stMetric"] {{ background:{PAPER}; border:1px solid {HAIR};
  border-top:2px solid {SAGE}; padding:14px 16px; }}
[data-testid="stMetricLabel"] p {{ text-transform:uppercase; letter-spacing:1.3px;
  font-size:0.6rem !important; color:{TAUPE}; }}
[data-testid="stMetricValue"] {{ font-family:{SERIF}; color:{INK}; font-size:1.5rem; }}
[data-testid="stDataFrame"] {{ border:1px solid {HAIR}; }}
[data-testid="stExpander"] {{ border:none !important; box-shadow:none !important; }}
[data-testid="stExpander"] summary p {{ text-transform:uppercase; letter-spacing:2px;
  font-size:0.7rem !important; color:{TAUPE}; }}

/* Top-level tab nav */
[data-baseweb="tab-list"] {{ gap:30px; border-bottom:1px solid {HAIR}; }}
[data-testid="stTabs"] button[role="tab"] {{ text-transform:uppercase; letter-spacing:2.4px;
  font-size:0.72rem; color:{TAUPE}; padding:6px 2px; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color:{SAGE}; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background:{SAGE} !important; }}

/* Inputs */
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label,
[data-testid="stDateInput"] label {{ text-transform:uppercase; letter-spacing:1.4px;
  font-size:0.62rem !important; color:{TAUPE}; }}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{ background:{PAPER};
  border-color:{HAIR}; border-radius:0; }}
[data-testid="stDateInput"] input {{ background:{PAPER}; border-radius:0; }}

/* Buttons (used by the chat shell): secondary = bordered serif tile; primary = link */
[data-testid="stBaseButton-secondary"] {{ background-color:{PAPER} !important;
  border:1px solid {HAIR} !important; border-radius:0 !important; box-shadow:none !important;
  color:{INK} !important; font-family:{SERIF}; font-size:1.0rem !important; line-height:1.4;
  text-align:left !important; padding:16px 18px !important; min-height:84px; white-space:normal; }}
[data-testid="stBaseButton-secondary"]:hover {{ border-color:{SAGE} !important; color:{SAGE} !important; }}
[data-testid="stBaseButton-secondary"] p {{ text-align:left; }}
[data-testid="stBaseButton-primary"] {{ background-color:transparent !important; border:none !important;
  box-shadow:none !important; color:{TAUPE} !important; text-transform:uppercase; letter-spacing:2px;
  font-size:0.7rem !important; text-decoration:underline; padding:0 !important; }}
[data-testid="stBaseButton-primary"]:hover {{ color:{INK} !important; }}

/* Chat message rules + input (kept identical to the original chat aesthetic) */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {{ display:none !important; }}
[data-testid="stChatMessage"] {{ background:transparent !important; padding:.15rem 0 .15rem 18px; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ border-left:2px solid {SAGE}; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{ border-left:2px solid {MUTED}; }}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{ color:{INK}; line-height:1.72; }}
[data-testid="stChatInput"] * {{ border:none !important; box-shadow:none !important; background-color:transparent !important; }}
[data-testid="stChatInput"] {{ border:none !important; border-bottom:1px solid {INK} !important;
  border-radius:0 !important; background-color:transparent !important; }}
[data-testid="stChatInput"] textarea {{ color:{INK} !important; }}
[data-testid="stChatInput"] textarea::placeholder {{ font-family:{SERIF}; font-style:italic; color:{MUTED} !important; }}
[data-testid="stChatInputSubmitButton"] {{ background-color:{INK} !important; border-radius:50% !important; }}
[data-testid="stChatInputSubmitButton"] svg {{ color:{CREAM} !important; fill:{CREAM} !important; }}

/* Source/chip helpers reused by the chat shell */
.gl-src {{ font-size:0.9rem; color:{TAUPE}; margin:.25rem 0; font-family:{SANS}; }}
.gl-src a {{ color:{INDIGO}; text-decoration:none; }}
.gl-src a:hover {{ text-decoration:underline; }}
.gl-chip {{ display:inline-block; text-transform:uppercase; letter-spacing:1.8px; font-size:0.64rem; margin-bottom:.55rem; }}
.gl-empty {{ font-family:{SERIF}; font-size:1.16rem; color:{TAUPE}; line-height:1.5; margin:.3rem 0 1.4rem; }}

/* Chat shell (Ask tab) — uses the same tokens so it's indistinguishable from the others */
.gl-headline {{ font-family:{SERIF}; font-weight:400; font-size:34px; line-height:1.1; color:{INK}; margin:.15rem 0 0; }}
.gl-subline {{ font-family:{SERIF}; font-weight:500; font-size:16px; color:{TAUPE}; margin:.4rem 0 0; }}
.gl-tilehead {{ display:flex; justify-content:space-between; align-items:center; margin:0 0 6px; }}
.gl-tilelabel {{ text-transform:uppercase; letter-spacing:1.8px; font-size:0.62rem; }}
.gl-arrow {{ color:{MUTED}; font-size:0.95rem; }}
.gl-footer {{ display:flex; justify-content:space-between; align-items:center; margin-top:16px; }}
.gl-foot-l {{ text-transform:uppercase; letter-spacing:2px; font-size:0.62rem; color:{MUTED}; }}
.gl-foot-r {{ text-transform:uppercase; letter-spacing:2px; font-size:0.62rem; color:{SAGE}; }}
</style>
"""


def inject_css():
    """Inject the single global stylesheet. Call once at the top of each tab."""
    st.markdown(_CSS, unsafe_allow_html=True)


# --- Layout primitives --------------------------------------------------------
def brand_bar(eyebrow):
    st.markdown(
        f'<div class="gl-brand"><span class="gl-leaf">🌿</span>'
        f'<span class="gl-word">GreenLeaf</span>'
        f'<span class="gl-eyebrow">{_html.escape(eyebrow)}</span></div>',
        unsafe_allow_html=True)


def section_header(label, title, sub=None):
    sub_html = f'<div class="gl-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="gl-eyebrow" style="margin-top:8px;">{_html.escape(label)}</div>'
        f'<div class="gl-h2">{title}</div>{sub_html}'
        f'<hr class="gl-hair">', unsafe_allow_html=True)


def kpi(value, label, sub=None, accent=SAGE):
    return {"value": value, "label": label, "sub": sub or "", "accent": accent}


def kpi_row(cards):
    cells = "".join(
        f'<div class="gl-kpi" style="border-top-color:{c["accent"]}">'
        f'<div class="gl-kpi-label">{_html.escape(c["label"])}</div>'
        f'<div class="gl-kpi-value">{c["value"]}</div>'
        f'<div class="gl-kpi-sub">{c["sub"]}</div></div>'
        for c in cards)
    st.markdown(f'<div class="gl-kpis">{cells}</div>', unsafe_allow_html=True)


def card_title(text):
    st.markdown(f'<div class="gl-cardtitle">{text}</div>', unsafe_allow_html=True)


def caveat_note(text):
    st.markdown(
        f'<div class="gl-caveat"><div class="gl-cardtitle">Caveat</div>'
        f'<p>{text}</p></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="gl-note">{text}</div>', unsafe_allow_html=True)


def hairline():
    st.markdown('<hr class="gl-hair">', unsafe_allow_html=True)


# --- Chart theme --------------------------------------------------------------
def style_fig(fig, height=300, legend=False, xtitle=None, ytitle=None, ygrid=True, xgrid=True):
    """Apply the unified chart theme to EVERY Plotly figure. Forest/earth on cream,
    hairline gridlines, clean margins, consistent fonts — never Plotly blue/neon."""
    fig.update_layout(
        height=height, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family=SANS, color=INK, size=12), colorway=CAT,
        margin=dict(l=10, r=14, t=10, b=10),
        showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color=TAUPE)),
        hoverlabel=dict(bgcolor=CREAM, font=dict(family=SANS, color=INK)),
    )
    fig.update_xaxes(gridcolor=HAIR, zerolinecolor=MUTED, linecolor=HAIR,
                     title_text=xtitle, title_font=dict(size=11, color=TAUPE),
                     tickfont=dict(size=10, color=TAUPE), showgrid=xgrid)
    fig.update_yaxes(gridcolor=HAIR, zerolinecolor=MUTED, linecolor=HAIR,
                     title_text=ytitle, title_font=dict(size=11, color=TAUPE),
                     tickfont=dict(size=10, color=TAUPE), showgrid=ygrid)
    return fig
