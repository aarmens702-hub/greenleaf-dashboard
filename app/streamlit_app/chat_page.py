"""Stage 4: Streamlit chat UI wrapping the Stage 3 answer() orchestrator.

Exposes render_chat(), which draws the full chat experience into the current
Streamlit page/tab. It holds conversation state in st.session_state.messages
(session-only; nothing is persisted to disk) and calls app.chat.answer.answer()
for each question. When the 4-tab dashboard is built, render_chat() will be
called from inside a tab instead of main.py.

Visual style: Lululemon-inspired editorial/lifestyle aesthetic — warm cream
page, serif headlines, hairline dividers, no shadows. Palette (reuse in other
tabs):
    cream  #F5F2EC   ink    #1F2A1F   taupe  #5A5A55   muted  #88897F
    hair   #D9D4C7   sage   #3D5A3D   terra  #8A5A3D   indigo #5A5A8A
The forest-sage accent (#3D5A3D) is also primaryColor in .streamlit/config.toml.
"""
import html
import traceback

import streamlit as st

from app.chat.answer import answer
from app.streamlit_app import theme as T

# Brand accent — forest sage. Match across other tabs.
ACCENT = "#3D5A3D"

# Sample questions, each tagged with an *illustrative* route label + meaning-coded
# color (display-only; the real route is decided at runtime by answer()/classify()).
SAMPLE_QUESTIONS = [
    {"q": "What's our overall precision benefit?", "label": "DATA", "color": "#3D5A3D"},
    {"q": "How do same-day responses compare to delayed ones?", "label": "DATA", "color": "#3D5A3D"},
    {"q": "Is our tomato yield above the BC benchmark?", "label": "DATA + RESEARCH", "color": "#8A5A3D"},
    {"q": "What does research say about strawberry fertigation?", "label": "RESEARCH", "color": "#5A5A8A"},
]

# Single bounded CSS block — editorial light aesthetic on Streamlit's light theme.
_CSS = """
<style>
:root{
  --cream:#F5F2EC; --ink:#1F2A1F; --taupe:#5A5A55; --muted:#88897F;
  --hair:#D9D4C7; --sage:#3D5A3D; --terra:#8A5A3D; --indigo:#5A5A8A;
}
/* Force warm cream even under a dark system theme; calm centered measure. */
.stApp { background-color:#F5F2EC !important; }
.block-container { padding:40px 44px 56px !important; max-width:880px; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background:#F5F2EC !important; }

/* Serif headline + subline. No 700 weight anywhere. */
.gl-headline { font-family:Georgia,'Times New Roman',serif; font-weight:400;
  font-size:40px; line-height:1.08; color:var(--ink); margin:0; }
.gl-subline  { font-family:Georgia,'Times New Roman',serif; font-weight:500;
  font-size:18px; color:var(--taupe); margin:.45rem 0 0; }

/* Tiny uppercase category labels — a key vibe element. */
.gl-eyebrow { text-transform:uppercase; letter-spacing:2.5px; font-size:0.7rem; color:var(--taupe); }
.gl-brand { display:flex; align-items:center; gap:10px; }
.gl-leaf { width:26px; height:26px; border-radius:50%; background:var(--sage); color:var(--cream);
  display:flex; align-items:center; justify-content:center; font-size:13px; }
.gl-hair { border:none; border-top:1px solid var(--hair); margin:18px 0 26px; }

/* CLEAR (the page's only `primary` button) = underlined uppercase link, no chrome.
   Streamlit 1.41 tags buttons data-testid="stBaseButton-<kind>". */
[data-testid="stBaseButton-primary"] {
  background-color:transparent !important; border:none !important; box-shadow:none !important;
  color:var(--taupe) !important; text-transform:uppercase; letter-spacing:2px;
  font-size:0.7rem !important; text-decoration:underline; padding:0 !important; }
[data-testid="stBaseButton-primary"]:hover { color:var(--ink) !important; }

/* Sample tiles = bordered serif cells (the page's `secondary` buttons). One border
   per element — no container wrappers — so nothing doubles or boxes other blocks. */
.gl-tilehead { display:flex; justify-content:space-between; align-items:center; margin:0 0 6px; }
.gl-tilelabel { text-transform:uppercase; letter-spacing:1.8px; font-size:0.62rem; }
.gl-arrow { color:var(--muted); font-size:0.95rem; }
[data-testid="stBaseButton-secondary"] {
  background-color:var(--cream) !important; border:1px solid var(--hair) !important;
  border-radius:0 !important; box-shadow:none !important; color:var(--ink) !important;
  font-family:Georgia,'Times New Roman',serif; font-size:1.02rem !important; line-height:1.4;
  text-align:left !important; padding:18px 20px !important; min-height:90px; white-space:normal; }
[data-testid="stBaseButton-secondary"]:hover {
  border-color:var(--sage) !important; color:var(--sage) !important; }
[data-testid="stBaseButton-secondary"] p { text-align:left; }

/* Chat messages: hide avatars, use a 2px left rule (sage=user, muted=assistant). */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { display:none !important; }
[data-testid="stChatMessage"] { background:transparent !important; padding:.15rem 0 .15rem 18px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { border-left:2px solid var(--sage); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { border-left:2px solid var(--muted); }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color:var(--ink); line-height:1.72; }

/* Expander labels: tiny editorial uppercase, no border box. */
[data-testid="stExpander"] { border:none !important; box-shadow:none !important; }
[data-testid="stExpander"] summary p {
  text-transform:uppercase; letter-spacing:2px; font-size:0.7rem !important; color:var(--taupe); }

/* Editorial detail classes. */
.gl-chip { display:inline-block; text-transform:uppercase; letter-spacing:1.8px;
  font-size:0.64rem; margin-bottom:.55rem; }
.gl-src { font-size:0.92rem; color:var(--taupe); margin:.25rem 0; }
.gl-src a { color:var(--indigo); text-decoration:none; }
.gl-src a:hover { text-decoration:underline; }
.gl-empty { font-family:Georgia,'Times New Roman',serif; font-size:1.18rem; color:var(--taupe);
  line-height:1.5; margin:.3rem 0 1.6rem; }

/* Chat input: not boxed — a single bottom-rule line, italic-serif placeholder.
   Nuke borders/fills on every nested node, then add one bottom rule on the shell. */
[data-testid="stChatInput"] * {
  border:none !important; box-shadow:none !important; background-color:transparent !important; }
[data-testid="stChatInput"] {
  border:none !important; border-bottom:1px solid var(--ink) !important; border-radius:0 !important;
  background-color:transparent !important; }
[data-testid="stChatInput"] textarea { color:var(--ink) !important; }
[data-testid="stChatInput"] textarea::placeholder {
  font-family:Georgia,'Times New Roman',serif; font-style:italic; color:var(--muted) !important; }
/* Send action = small dark circle (rule comes after `*` so it wins on source order). */
[data-testid="stChatInputSubmitButton"] {
  background-color:var(--ink) !important; border-radius:50% !important; }
[data-testid="stChatInputSubmitButton"] svg { color:var(--cream) !important; fill:var(--cream) !important; }

/* Footer line. */
.gl-footer { display:flex; justify-content:space-between; align-items:center; margin-top:16px; }
.gl-foot-l { text-transform:uppercase; letter-spacing:2px; font-size:0.62rem; color:var(--muted); }
.gl-foot-r { text-transform:uppercase; letter-spacing:2px; font-size:0.62rem; color:var(--sage); }
</style>
"""


def _route_meta(routing):
    """Map a routing dict to a (label, color) editorial chip for the REASONING expander."""
    needs_data = (routing or {}).get("needs_data")
    needs_docs = (routing or {}).get("needs_docs")
    if needs_data and needs_docs:
        return "DATA + RESEARCH", "#8A5A3D"
    if needs_data:
        return "DATA", "#3D5A3D"
    if needs_docs:
        return "RESEARCH", "#5A5A8A"
    return "DIRECT", "#5A5A55"


def escape_dollars(text):
    """Escape literal '$' so st.markdown won't read e.g. "$47,033.53" as LaTeX math.

    Apply ONLY to assistant answer text — never to source lines or routing JSON.
    Leaves markdown bold/lists intact; just neutralizes the math delimiter.
    """
    return (text or "").replace("$", "\\$")


def _render_sources(sources):
    """Render the 'SOURCES · n' expander for an assistant message."""
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)}"):
        for s in sources:
            if s.get("type") == "doc":
                title = html.escape(s.get("title") or "Untitled")
                url = html.escape(s.get("source_url") or "", quote=True)
                st.markdown(
                    f'<div class="gl-src">{title} — '
                    f'<a href="{url}" target="_blank">{url}</a></div>',
                    unsafe_allow_html=True,
                )
            elif s.get("type") == "data":
                fn = s.get("function") or "unknown"
                args = s.get("args") or {}
                arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                st.markdown(
                    f'<div class="gl-src"><code>{html.escape(fn)}'
                    f'({html.escape(arg_str)})</code></div>',
                    unsafe_allow_html=True,
                )


def _render_routing(trace):
    """Render the 'REASONING' expander: an editorial route chip + raw routing dict."""
    routing = (trace or {}).get("routing")
    if not routing:
        return
    with st.expander("Reasoning"):
        label, color = _route_meta(routing)
        st.markdown(f'<span class="gl-chip" style="color:{color};">{label}</span>', unsafe_allow_html=True)
        st.json(routing)


def _render_assistant(msg):
    """Render a single assistant message: answer, sources, routing, and any error."""
    st.markdown(escape_dollars(msg["content"]))
    _render_sources(msg.get("sources", []))
    _render_routing(msg.get("trace", {}))
    if msg.get("traceback"):
        with st.expander("Error details"):
            st.code(msg["traceback"])


def _handle_question(question):
    """Append the user turn, call answer(), append the assistant turn, rerun."""
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Thinking..."):
            result = answer(question)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "trace": result.get("trace", {}),
        })
    except Exception as e:  # noqa: BLE001 — UI must never crash; show the error instead.
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"I hit an error: {type(e).__name__}. "
                "Check the .env / network and try again."
            ),
            "sources": [],
            "trace": {},
            "traceback": traceback.format_exc(),
        })
    st.rerun()


def render_chat() -> None:
    """Render the GreenLeaf chat UI into the current Streamlit page/tab."""
    T.inject_css()  # shared design system (supersedes the legacy local _CSS below)

    # Top brand bar: leaf mark + uppercase wordmark on the left, CLEAR link on the right.
    bar_l, bar_r = st.columns([6, 1])
    with bar_l:
        st.markdown(
            '<div class="gl-brand"><span class="gl-leaf">🌿</span>'
            '<span class="gl-eyebrow">GREENLEAF · ANALYST</span></div>',
            unsafe_allow_html=True,
        )
    with bar_r:
        # type="primary" tags this as the page's only `primary` button so CSS can
        # style it as a link without touching the `secondary` sample tiles.
        if st.button("Clear", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Serif headline + muted serif accent line, then a single hairline divider.
    st.markdown(
        '<div class="gl-headline">Ask GreenLeaf</div>'
        '<div class="gl-subline">Ask about GreenLeaf\'s data or BC agriculture context.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="gl-hair">', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Empty state: a calm editorial welcome line.
    if not st.session_state.messages:
        st.markdown(
            '<div class="gl-empty">Welcome. Ask about yields, costs, pest response, or BC '
            'agriculture research — every answer cites where it came from.</div>',
            unsafe_allow_html=True,
        )

    # Conversation history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                _render_assistant(msg)

    # Sample-question grid: 2×2 product tiles. A button returns True only on the
    # rerun where it was clicked, so we capture the seeded question like typed input.
    question = None
    st.markdown('<div class="gl-eyebrow" style="margin-bottom:12px;">SUGGESTED · 4 PROMPTS</div>',
                unsafe_allow_html=True)
    for row_start in (0, 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            s = SAMPLE_QUESTIONS[row_start + j]
            with col:
                # Route label + arrow sit just above the bordered question tile
                # (the `secondary` button). One bordered element per tile — no
                # container wrappers — so nothing doubles.
                st.markdown(
                    f'<div class="gl-tilehead">'
                    f'<span class="gl-tilelabel" style="color:{s["color"]};">{s["label"]}</span>'
                    f'<span class="gl-arrow">↗</span></div>',
                    unsafe_allow_html=True,
                )
                if st.button(s["q"], key=f"sample_{row_start + j}", use_container_width=True):
                    question = s["q"]

    typed = st.chat_input("Ask about GreenLeaf — try data, research, or both")
    if typed:
        question = typed

    # Footer line: provenance on the left, live indicator on the right.
    st.markdown(
        '<div class="gl-footer">'
        '<span class="gl-foot-l">SOURCED · RETRIEVAL OVER 9 DOCS + 10 DATA FUNCTIONS</span>'
        '<span class="gl-foot-r">● LIVE</span></div>',
        unsafe_allow_html=True,
    )

    if question:
        _handle_question(question)
