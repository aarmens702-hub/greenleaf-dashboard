"""Stage 3 (v2): an agentic tool-calling loop for the GreenLeaf chat.

Replaces the single-shot router. The model is given tools (document search, the
five fixed analytics functions, and a flexible parameterized query_data) and runs
a gather→look→adjust loop until it can answer, then returns a cited answer.

Public entry point: answer(question, max_iterations=6) -> dict. app/chat/answer.py
re-exports it so existing imports (the Streamlit UI) keep working.

LLM provider: DeepSeek chat completions (OpenAI-compatible function calling). The
only provider-specific code is _chat() below; it reuses router's key/base-url
plumbing. Tool results are appended to the messages array as role="tool" turns.
"""
import json

import pandas as pd
import httpx

from app.chat import router
from app.chat import data_queries as dq
from app.rag.retrieve import retrieve

AGENT_MODEL = "deepseek-chat"
_TRUNC = 4000  # cap each tool result fed back to the model

# ---------------------------------------------------------------------------
# query_data — the flexible, SAFE escape hatch (parameterized pandas only).
# ---------------------------------------------------------------------------

# Whitelisted tables (callables so we always read the live module globals).
_TABLES = {
    "season": lambda: dq.SEASON,
    "daily": lambda: dq.DAILY,
    "actions": lambda: dq.ACTIONS,
    "scouting": lambda: dq.SCOUTING,
    "plot_meta": lambda: dq.PLOT_META,
}
_AGG_FUNCS = {"mean", "sum", "count", "min", "max", "median"}
_JOIN_COLS = ("crop", "treatment")
_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "contains"}


def _coerce_val(series, val):
    """Coerce a filter value to match the column dtype (e.g. date strings)."""
    if pd.api.types.is_datetime64_any_dtype(series) and isinstance(val, str):
        return pd.Timestamp(val)
    return val


def _eq_mask(series, val):
    """Equality mask; case-insensitive for string columns."""
    if series.dtype == object and isinstance(val, str):
        return series.astype(str).str.lower() == val.lower()
    return series == _coerce_val(series, val)


def _apply_filter(df, col, cond):
    """Apply one column filter. cond may be a scalar (==), list (in), or an
    operator dict like {">=": x, "<=": y} / {"in": [...]} / {"contains": "txt"}.
    All comparisons are parameterized pandas — no eval, no query strings."""
    series = df[col]
    if isinstance(cond, dict):
        mask = pd.Series(True, index=df.index)
        for op, val in cond.items():
            if op not in _OPS:
                raise ValueError(f"unsupported filter op '{op}' on '{col}'")
            if op == "==":
                mask &= _eq_mask(series, val)
            elif op == "!=":
                mask &= ~_eq_mask(series, val)
            elif op == ">":
                mask &= series > _coerce_val(series, val)
            elif op == ">=":
                mask &= series >= _coerce_val(series, val)
            elif op == "<":
                mask &= series < _coerce_val(series, val)
            elif op == "<=":
                mask &= series <= _coerce_val(series, val)
            elif op == "in":
                vals = val if isinstance(val, list) else [val]
                mask &= series.isin(vals)
            elif op == "contains":
                mask &= series.astype(str).str.contains(str(val), case=False, na=False)
        return df[mask]
    if isinstance(cond, list):
        return df[series.isin(cond)]
    return df[_eq_mask(series, cond)]


def _records(df):
    """DataFrame -> JSON-safe list of dicts (numpy/Timestamp coerced, floats rounded)."""
    out = []
    for rec in df.to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp):
                v = v.strftime("%Y-%m-%d")
            elif isinstance(v, float):
                v = round(v, 4)
            elif hasattr(v, "item"):  # numpy scalar
                v = v.item()
            clean[str(k)] = v
        out.append(clean)
    return out


def _add_per_plot(result_df, group_by, df, target_cols):
    """Append '<col>_per_plot' = col / distinct plots in that group, for the given
    count/sum columns. Per-plot is a fair, size-unbiased basis for ranking groups
    of different sizes. Returns (df, added_any)."""
    if not target_cols or "plot_id" not in df.columns:
        return result_df, False
    counts = df.groupby(group_by)["plot_id"].nunique().reset_index(name="_n_plots")
    result_df = result_df.merge(counts, on=group_by, how="left")
    for c in target_cols:
        result_df[f"{c}_per_plot"] = (result_df[c] / result_df["_n_plots"]).round(4)
    return result_df.drop(columns="_n_plots"), True


def query_data(table, filters=None, group_by=None, aggregate=None,
               sort_by=None, ascending=False, limit=20, time_bucket=None):
    """Flexible filtered/grouped aggregation over a whitelisted GreenLeaf table.

    Safe by construction: only parameterized pandas, whitelisted tables and
    aggregate functions, and column-name validation. crop/treatment are auto-joined
    from plot_meta when referenced but absent. time_bucket ('week'|'month') resamples
    by date for trend questions and disables the row cap so the whole season is
    returned. Returns {table, n_rows_matched, result, columns_used, notes}.
    """
    if table not in _TABLES:
        raise ValueError(
            f"Table '{table}' not available. Valid tables: {', '.join(sorted(_TABLES))}."
        )
    df = _TABLES[table]().copy()
    notes = []

    filters = filters or {}
    if isinstance(group_by, str):
        group_by = [group_by]
    group_by = list(group_by or [])
    aggregate = dict(aggregate or {})

    # Time bucketing: resample by date into week/month periods so 'over the season'
    # questions see the full range (the prior bug truncated to the first 20 dates).
    if time_bucket:
        if time_bucket not in ("week", "month"):
            raise ValueError("time_bucket must be 'week' or 'month'.")
        if "date" not in df.columns:
            raise ValueError(
                f"time_bucket needs a 'date' column; table '{table}' has none. "
                f"Use 'daily', 'actions', or 'scouting'."
            )
        freq = "W" if time_bucket == "week" else "M"
        df["period"] = df["date"].dt.to_period(freq).dt.start_time
        if "period" not in group_by:
            group_by = ["period", *group_by]
        if not sort_by:
            sort_by, ascending = "period", True  # chronological by default

    # Auto-join plot_meta for crop/treatment if referenced anywhere but missing.
    referenced = set(filters) | set(group_by) | set(aggregate)
    if sort_by:
        referenced.add(sort_by)
    need_join = [c for c in _JOIN_COLS if c in referenced and c not in df.columns]
    if need_join and "plot_id" in df.columns:
        df = df.merge(dq.PLOT_META[["plot_id", *_JOIN_COLS]], on="plot_id", how="left")
        notes.append(f"auto-joined plot_meta for {need_join}")
    cols = set(df.columns)

    # Validate referenced INPUT columns up front with an actionable error, so the
    # model fixes a typo'd column in ONE retry instead of guessing. (sort_by is
    # checked later — it may name an aggregate output rather than an input column.)
    unknown = sorted(c for c in set(filters) | set(group_by) | set(aggregate) if c not in cols)
    if unknown:
        raise ValueError(
            f"Column(s) {unknown} not found in table '{table}'. "
            f"Available columns: {sorted(cols)}."
        )
    bad_funcs = sorted({f for f in aggregate.values() if f not in _AGG_FUNCS})
    if bad_funcs:
        raise ValueError(
            f"Aggregate function(s) {bad_funcs} not supported. "
            f"Use one of: {', '.join(sorted(_AGG_FUNCS))}."
        )

    # Filters (all columns now known to exist).
    for col, cond in filters.items():
        df = _apply_filter(df, col, cond)
    n_matched = int(len(df))

    # Column-collision fix: an aggregate target that is also a group_by key cannot be
    # aggregated by pandas. Drop it with a clear note (a count-style request still
    # succeeds via the size() fallback below) rather than raising and forcing a retry.
    valid_agg = {}
    for col, func in aggregate.items():
        if col in group_by:
            notes.append(
                f"Cannot aggregate column '{col}' because it is also the group_by key; "
                f"dropped it. Choose a different column to aggregate, or remove group_by."
            )
        else:
            valid_agg[col] = func

    # Shape the result.
    if valid_agg and group_by:
        result_df = df.groupby(group_by).agg(valid_agg).reset_index()
        # Size-bias guard: count/sum totals favour groups with more plots. Add a
        # per-plot version of each so rankings/comparisons aren't size-biased.
        norm_cols = [c for c, f in valid_agg.items() if f in ("count", "sum")]
        result_df, added = _add_per_plot(result_df, group_by, df, norm_cols)
        if added:
            notes.append("added *_per_plot columns; rank on these, not raw count/sum (size-biased)")
    elif valid_agg:
        result_df = df.agg(valid_agg).to_frame().T.reset_index(drop=True)
    elif group_by:
        result_df = df.groupby(group_by).size().reset_index(name="count")
        result_df, added = _add_per_plot(result_df, group_by, df, ["count"])
        if added:
            notes.append("added count_per_plot (rows per plot); rank on this, not raw count (size-biased)")
    else:
        result_df = df

    # Sort.
    if sort_by and sort_by in result_df.columns:
        result_df = result_df.sort_values(sort_by, ascending=bool(ascending))
    elif sort_by:
        notes.append(f"skipped sort on unknown column '{sort_by}'")

    # Limit — but NEVER silently truncate a time-series: a time-bucketed result, or
    # one ordered by date/period, returns the full ordered series so trends are whole.
    n_result = len(result_df)
    is_timeseries = bool(time_bucket) or sort_by in ("date", "period")
    if is_timeseries:
        notes.append(f"time-series: returned all {n_result} buckets (no row cap)")
    else:
        eff_limit = max(1, min(int(limit), 200))
        if n_result > eff_limit:
            notes.append(f"rows_returned={eff_limit} of rows_matched-groups={n_result} (truncated)")
        result_df = result_df.head(eff_limit)

    return {
        "table": table,
        "n_rows_matched": n_matched,
        "result": _records(result_df),
        "columns_used": list(result_df.columns),
        "notes": "; ".join(notes) if notes else "ok",
    }


# ---------------------------------------------------------------------------
# Tools: thin wrappers around existing code + their JSON schemas.
# ---------------------------------------------------------------------------

_FIXED_TOOLS = {
    "precision_benefit_summary": dq.precision_benefit_summary,
    "response_time_payoff": dq.response_time_payoff,
    "treatment_summary": dq.treatment_summary,
    "top_at_risk_plots": dq.top_at_risk_plots,
    "greenleaf_vs_benchmark": dq.greenleaf_vs_benchmark,
    # P0: previously-unexposed but tested functions, now first-class tools.
    "season_overview": dq.season_overview,
    "avg_profit_by_crop": dq.avg_profit_by_crop,
    "plot_status": dq.plot_status,
    "cost_breakdown": dq.cost_breakdown,
    "alert_response_summary": dq.alert_response_summary,
}

_QUERY_DATA_KEYS = {"table", "filters", "group_by", "aggregate", "sort_by",
                    "ascending", "limit", "time_bucket"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the BC agriculture knowledge base (9 documents on greenhouse "
                "industry, precision-ag research, energy costs, financing, crops, pests, "
                "labor, climate, input prices) for background/context/research. Use for "
                "'what does research say', benchmarks, industry context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "k": {"type": "integer", "description": "Number of chunks to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "precision_benefit_summary",
            "description": "GreenLeaf's total/per-plot precision benefit and its share of profit.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "response_time_payoff",
            "description": "How stress recovery varies by how fast the team responded to alerts (action delay days).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "treatment_summary",
            "description": "Yield, marketable ratio, profit, ROI per treatment, optionally filtered by crop. Includes sample sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string", "description": "Optional crop filter, e.g. 'Strawberry'."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_at_risk_plots",
            "description": "Plots ranked by stress/spike risk for a given 7-day week.",
            "parameters": {
                "type": "object",
                "properties": {
                    "week_start": {"type": "string", "description": "Monday of the week, 'YYYY-MM-DD'."},
                    "k": {"type": "integer", "description": "How many plots to return (default 5)."},
                },
                "required": ["week_start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "greenleaf_vs_benchmark",
            "description": "GreenLeaf's value vs an external BC benchmark, normalized. Known metric_name: 'greenhouse_tomato_annual_yield'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {"type": "string", "description": "Benchmark metric name."},
                },
                "required": ["metric_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "season_overview",
            "description": "Whole-operation summary: farm/greenhouse/plot/crop counts and season totals (revenue, cost, profit, overall ROI, mean marketable ratio, mean yield). Use for 'overview', 'how is the whole operation doing', totals/ROI.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "avg_profit_by_crop",
            "description": "Mean season profit per plot for each crop. Use for 'which crop is most/least profitable', per-crop profit comparisons.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_status",
            "description": "Single plot snapshot: metadata, season financials, and recent daily readings (stress, alerts, actions). Use for 'how is plot P0001 doing', any question about ONE specific plot id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_id": {"type": "string", "description": "Plot id, e.g. 'P0001'."},
                    "last_n_days": {"type": "integer", "description": "How many recent daily rows to include (default 14)."},
                },
                "required": ["plot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cost_breakdown",
            "description": "Mean daily input-cost components (energy, fertilizer, labor, pesticide, water, total) per plot-day, optionally filtered by crop/treatment. This is the CANONICAL source for cost-mix / 'biggest cost driver' questions — do NOT derive cost breakdowns from the actions table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string", "description": "Optional crop filter."},
                    "treatment": {"type": "string", "description": "Optional treatment filter."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alert_response_summary",
            "description": "How alerts were handled: alert volume, action rate, mean response delay (days), and mean 3-day stress recovery, optionally filtered by crop. Use for alert/response/recovery questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string", "description": "Optional crop filter."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": (
                "Run a flexible filtered/grouped aggregation over GreenLeaf's raw data "
                "tables. Use for specific questions the other tools don't cover, e.g. "
                "'average profit for strawberry shade plots', 'which crop has highest "
                "stress'. Tables: season (per-plot financials+yield), daily (per-plot-"
                "per-day sensors/stress/costs), actions (input applications), scouting "
                "(inspections), plot_meta (plot crop/treatment/greenhouse/farm). crop and "
                "treatment are available on all via auto-join.\n"
                "Key columns — season: crop, treatment, season_profit_cad, season_roi, "
                "season_yield_kg_m2, marketable_ratio, season_revenue_cad, total_cost_cad, "
                "precision_benefit_cad, region; daily: date, crop, treatment, "
                "plant_stress_index, spike_3d, alert_flag, alert_type, action_taken, "
                "action_delay_days, vpd_kpa, daily_total_input_cost; actions: input_type, "
                "total_cost, is_precision_action, purpose; scouting: pest_severity, "
                "disease_severity, blossom_count; plot_meta: crop, treatment, greenhouse_id, "
                "farm_id, region.\n"
                "filters: {column: value} (equality, case-insensitive for text), "
                "{column: [v1,v2]} (in), or {column: {\">=\": x, \"<=\": y}} / {\">\": n} / "
                "{\"contains\": \"text\"}. group_by: list of columns. aggregate: "
                "{column: func} where func in mean,sum,count,min,max,median. sort_by + "
                "ascending + limit control ordering.\n"
                "time_bucket ('week' or 'month'): for ANY 'over time / over the season / "
                "trend' question, set this (on daily/actions/scouting) to resample by date "
                "so you get the WHOLE season, not the first 20 days. It returns one row per "
                "bucket (a 'period' column), chronologically, with no truncation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "enum": sorted(_TABLES)},
                    "filters": {"type": "object", "description": "Column filters (see description)."},
                    "group_by": {"type": "array", "items": {"type": "string"}},
                    "aggregate": {"type": "object", "description": "{column: aggfunc}."},
                    "sort_by": {"type": "string"},
                    "ascending": {"type": "boolean"},
                    "limit": {"type": "integer"},
                    "time_bucket": {"type": "string", "enum": ["week", "month"],
                                    "description": "Resample by date for trend/over-season questions."},
                },
                "required": ["table"],
            },
        },
    },
]


def _execute_tool(name, args):
    """Run a tool. Returns (result, status). Never raises — errors become the
    tool result so the agent can read them and adjust."""
    try:
        if name == "search_documents":
            chunks = retrieve(args.get("query", ""), k=int(args.get("k", 5) or 5))
            res = [
                {"title": c.get("title"), "source_url": c.get("source_url"),
                 "text": (c.get("text") or "")[:600]}
                for c in chunks
            ]
            return res, "ok"
        if name == "query_data":
            clean = {k: v for k, v in (args or {}).items() if k in _QUERY_DATA_KEYS}
            return query_data(**clean), "ok"
        if name in _FIXED_TOOLS:
            out = _FIXED_TOOLS[name](**(args or {}))
            if isinstance(out, pd.DataFrame):
                out = {"rows": _records(out)}
            return out, "ok"
        return f"Unknown tool '{name}'.", "error"
    except Exception as e:  # noqa: BLE001 — surface the error to the agent, don't crash.
        return f"{type(e).__name__}: {e}", "error"


def _tool_content(result):
    """Serialize a tool result to a string for the messages array (truncated)."""
    if isinstance(result, str):
        return result[:_TRUNC]
    try:
        return json.dumps(result, default=str)[:_TRUNC]
    except (TypeError, ValueError):
        return str(result)[:_TRUNC]


# ---------------------------------------------------------------------------
# Agent loop.
# ---------------------------------------------------------------------------

AGENT_SYSTEM = """You are GreenLeaf's analytics agent for a BC controlled-environment agriculture operator. You have tools to query GreenLeaf's data and search a BC agriculture knowledge base. Your job is to ACTUALLY ANSWER the user's question.

How to work:
1. Understand what the user is really asking. Infer intent even if phrased loosely.
2. Call whatever tools you need to gather evidence. You can call multiple tools and call them repeatedly.
3. LOOK at each result. If it didn't give you what you need, ADJUST — try a different tool, a different table, a broader filter, or a different grouping. Don't give up after one attempt.
4. Once you have enough to answer, give a clear, specific answer with numbers.

Tool selection priority:
1. PREFER the specific analytics tools whenever they directly answer the question — they are authoritative and pre-computed correctly. Which tool owns which domain:
   - whole-operation totals / ROI / counts -> season_overview
   - per-crop profit ('which crop is most profitable') -> avg_profit_by_crop
   - a single specific plot (e.g. P0001) -> plot_status
   - cost mix / 'biggest cost driver' / cost components -> cost_breakdown
   - alert volume / response speed / recovery -> alert_response_summary
   - per-treatment yield/profit/ROI -> treatment_summary
   - precision benefit ($ / share of profit) -> precision_benefit_summary
   - response speed vs stress recovery -> response_time_payoff
   - riskiest week / at-risk plots -> top_at_risk_plots
   - GreenLeaf vs external benchmark -> greenleaf_vs_benchmark
2. Use query_data ONLY for slices no fixed tool covers (e.g. 'profit for strawberry shade plots specifically', 'which crop-treatment combo has highest stress'). Do not use query_data to re-derive something a fixed tool already returns.
3. Use search_documents for research, benchmarks, and background context.
Reach for the fewest tools that answer the question. Most questions need 1-2 tool calls.

For any 'over time / over the season / trend' question, use query_data with time_bucket ('week' or 'month') so you see the WHOLE season, not the first 20 days.

For cost questions, use cost_breakdown. The daily cost columns are the cost MIX; season total_cost_cad is the P&L total. Do not derive cost breakdowns from the actions table.

Before saying you cannot answer, you MUST try other tools. If one tool's output lacks what you need, call a different tool — query_data on the season table has revenue, cost, profit, and precision_benefit per plot. Only refuse after genuinely exhausting the tools.

When answering 'riskiest', 'most', 'highest', or any ranking/comparison across groups (crops, treatments, etc.), use NORMALIZED measures — means, rates, or per-plot/per-day figures — NOT raw totals. Raw counts are biased toward groups that simply have more plots or more observations. If you compute a total, also compute the per-plot or mean version and rank on that. State which basis you used.

When asked about 'financial gain' relative to performance, the most defensible figure we have is precision_benefit (the measured dollar benefit of precision management, from precision_benefit_summary). Prefer reporting that as the concrete financial gain, and treat benchmark-yield-to-dollars as a clearly-flagged estimate (greenleaf_vs_benchmark returns a 'financial_gain' block for the tomato yield metric).

Rules for the final answer:
- Your FINAL answer must begin directly with the answer. Never open with process narration ("I notice", "Now I have", "Based on my analysis", "Let me", "Here's", "Looking at"). Just answer.
- Example of a BAD opener: "Here's the comparison of High Light vs Shade: High Light wins...". Example of a GOOD opener: "High Light outperforms Shade on profit ($839 vs $379)...". Start with the finding itself, never a framing phrase.
- LEAD WITH THE ANSWER and the numbers. Caveats come after, briefly.
- DO THE MATH yourself — comparisons, rankings, normalizations, percentages.
- NEVER suggest follow-up questions ("consider asking...", "you could rephrase...", "please provide..."). NEVER output a "What you can ask instead" section, a list of suggested questions, or "you would need to ask about X". This is strictly forbidden. If you truly cannot answer after trying multiple tools, say so in ONE sentence and stop — no suggestions, no bullet list of alternatives.
- Respect sample sizes: if a crop/treatment group has few plots (n<5), answer but note the small sample.
- Cite sources inline: documents by source_url, data by tool name in brackets e.g. [treatment_summary] or [query_data: season].
- Under 200 words unless asked for more. Plain language for farmers and lenders."""

_FORCE_FINAL_NUDGE = (
    "You have reached the tool-call limit. Do not call any more tools. Answer the "
    "question now using the evidence already gathered, and briefly note that the "
    "search was cut off if anything is incomplete."
)


def _chat(messages, tools=None, max_tokens=1024):
    """One DeepSeek chat-completions call. Returns the assistant message dict
    ({role, content, tool_calls?}). SWAP POINT for the provider."""
    if not router.USE_LIVE_API:
        raise RuntimeError(
            "No LLM API key is set, so live calls are disabled. Get a key at "
            "https://platform.deepseek.com and export DEEPSEEK_API_KEY."
        )
    payload = {"model": AGENT_MODEL, "messages": messages, "max_tokens": max_tokens, "stream": False}
    if tools is not None:
        payload["tools"] = tools
    resp = httpx.post(
        f"{router.DEEPSEEK_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {router.API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def _assistant_entry(msg):
    """Normalize an assistant message for the history (keep tool_calls if present)."""
    entry = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        entry["tool_calls"] = msg["tool_calls"]
    return entry


def _parse_args(raw):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def answer(question, max_iterations=6):
    """Answer a question with an agentic tool-calling loop.

    Args: question (str); max_iterations (int) — hard cap on model turns.
    Returns: {answer, sources, trace:{iterations, tool_calls, hit_max}}.
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    tool_calls_trace = []      # {name, args, status}
    data_sources = []          # {type:data, tool, function, args}
    doc_candidates = []        # {title, source_url}
    seen_data = set()
    seen_docs = set()

    iterations = 0
    hit_max = False
    final_text = None

    while iterations < max_iterations:
        iterations += 1
        msg = _chat(messages, tools=TOOL_SCHEMAS)
        messages.append(_assistant_entry(msg))
        calls = msg.get("tool_calls") or []
        if not calls:
            final_text = msg.get("content") or ""
            break
        for call in calls:
            fn = call.get("function", {}) or {}
            name = fn.get("name", "")
            args = _parse_args(fn.get("arguments"))
            result, status = _execute_tool(name, args)
            tool_calls_trace.append({"name": name, "args": args, "status": status})
            if status == "ok" and name == "search_documents" and isinstance(result, list):
                for c in result:
                    url = c.get("source_url") or ""
                    key = url or c.get("title")
                    if key and key not in seen_docs:
                        seen_docs.add(key)
                        doc_candidates.append({"title": c.get("title"), "source_url": url})
            elif status == "ok" and name != "search_documents":
                key = (name, json.dumps(args, sort_keys=True, default=str))
                if key not in seen_data:
                    seen_data.add(key)
                    data_sources.append({"type": "data", "tool": name, "function": name, "args": args})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": _tool_content(result),
            })
    else:
        # Loop exhausted without a final answer: make one forced answer call (no tools).
        hit_max = True
        messages.append({"role": "user", "content": _FORCE_FINAL_NUDGE})
        msg = _chat(messages, tools=None)
        final_text = msg.get("content") or "I can't answer that from GreenLeaf's data."

    if final_text is None:
        final_text = "I can't answer that from GreenLeaf's data."

    # Prefer doc sources actually cited (url in answer); else show what was searched.
    cited = [d for d in doc_candidates if d["source_url"] and d["source_url"] in final_text]
    docs = cited if cited else doc_candidates
    sources = [{"type": "doc", **d} for d in docs] + data_sources

    return {
        "answer": final_text,
        "sources": sources,
        "trace": {
            "iterations": iterations,
            "tool_calls": tool_calls_trace,
            "hit_max": hit_max,
        },
    }
