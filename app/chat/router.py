"""Stage 3 router: classify a question and decide what context to gather.

Uses the DeepSeek chat API (cheap/fast, OpenAI-compatible) to map a user question
onto the 10 data functions in data_queries.py and/or the document knowledge base,
returning strict JSON. All model calls now go through the shared seam in
app/chat/llm.py (retry/timeout/fallback/usage capture); this module just shapes the
classification prompt and parses the JSON.
"""
import json
import logging
import re

from app.chat import llm
from app.core import config

logger = logging.getLogger(__name__)

# Back-compat re-exports: provider plumbing now lives in app/chat/llm.py.
DEEPSEEK_BASE_URL = config.DEEPSEEK_BASE_URL
ROUTER_MODEL = config.CHAT_MODEL
API_KEY = llm.API_KEY
USE_LIVE_API = llm.USE_LIVE_API

if not USE_LIVE_API:
    logger.warning(
        "No LLM API key set — live calls disabled. Export DEEPSEEK_API_KEY "
        "(https://platform.deepseek.com) to enable router/answer calls."
    )


def _call_llm(model, system, user, max_tokens=1024):
    """Run the classifier prompt through the shared LLM seam and return text.

    SWAP POINT lives in app/chat/llm.py now; this wraps it so classify() is unchanged.
    """
    result = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        model=model,
    )
    return result.message.get("content") or ""


ROUTER_SYSTEM = """You are the routing layer for GreenLeaf's analytics assistant (a BC controlled-environment agriculture operator). Given a user question, decide what context is needed to answer it: computed data functions and/or a document knowledge base.

Available DATA FUNCTIONS (name — description — args):
1. season_overview — whole-operation totals: counts of farms/greenhouses/plots/crops, revenue, cost, profit, ROI, marketable ratio, yield. No args.
2. precision_benefit_summary — total $ benefit attributed to precision management and its share of total profit. No args.
3. response_time_payoff — how short-run stress recovery varies with how fast the team responded (by action delay days). No args.
4. avg_profit_by_crop — mean season profit per plot for each crop. No args.
5. treatment_summary — per-treatment table (yield, marketable ratio, profit, ROI). Optional arg: crop (str).
6. plot_status — one plot's metadata, season financials, and recent daily readings. REQUIRED arg: plot_id (str, e.g. "P0001"). Optional: last_n_days (int).
7. top_at_risk_plots — riskiest plots in a 7-day window. REQUIRED arg: week_start (str "YYYY-MM-DD"). Optional: k (int).
8. greenleaf_vs_benchmark — compare a GreenLeaf metric to an external benchmark. REQUIRED arg: metric_name (str). Known values: "greenhouse_tomato_annual_yield".
9. cost_breakdown — mean daily input cost components. Optional args: crop (str), treatment (str).
10. alert_response_summary — alert volume, action rate, response speed, recovery. Optional arg: crop (str).

Use the DOCUMENT knowledge base (needs_docs=true) for general agronomic / industry / research / climate / financing questions (research evidence, BC climate, crop characteristics, pest/disease management, energy/fertilizer/labor context, the precision-ag loan program). Use DATA functions (needs_data=true) for questions about GreenLeaf's own measured numbers. A question may need both.

RULES:
- Only choose functions whose REQUIRED args you can extract from the question. If a required arg (plot_id, week_start, metric_name) is missing, DO NOT include that function.
- Put extracted kwargs in data_args, keyed by function name. Use {} for functions that take no args.
- Respond with ONLY a JSON object — no markdown, no prose — in exactly this shape:
{"needs_docs": bool, "needs_data": bool, "data_functions": ["fn", ...], "data_args": {"fn": {"kwarg": value}}, "reasoning": "one short sentence"}"""

_FALLBACK = {
    "needs_docs": True,
    "needs_data": False,
    "data_functions": [],
    "data_args": {},
    "reasoning": "fallback: invalid router JSON",
}


def _coerce(d):
    if not isinstance(d, dict):
        return None
    return {
        "needs_docs": bool(d.get("needs_docs", True)),
        "needs_data": bool(d.get("needs_data", False)),
        "data_functions": list(d.get("data_functions", []) or []),
        "data_args": dict(d.get("data_args", {}) or {}),
        "reasoning": str(d.get("reasoning", "")),
    }


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return _coerce(json.loads(m.group(0)))
    except json.JSONDecodeError:
        return None


def classify(question):
    """Classify a question into routing instructions.

    Args: question (str).
    Returns: dict with keys needs_docs, needs_data, data_functions, data_args, reasoning.
    On invalid JSON, retries once; if still invalid, returns a docs-only fallback.
    Example: classify("which crop pays best?")["needs_data"] -> True
    """
    user = f"User question: {question}\n\nReturn ONLY the JSON object."
    parsed = _extract_json(_call_llm(ROUTER_MODEL, ROUTER_SYSTEM, user, max_tokens=600))
    if parsed is not None:
        return parsed
    reprompt = (
        "Your previous response was not valid JSON. Return ONLY the JSON object, "
        "no markdown, no prose, matching the required shape."
    )
    parsed = _extract_json(_call_llm(ROUTER_MODEL, ROUTER_SYSTEM, reprompt, max_tokens=600))
    return parsed if parsed is not None else dict(_FALLBACK)
