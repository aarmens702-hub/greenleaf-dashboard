# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project-Specific Guidelines: GreenLeaf Dashboard

### Project context
This is a Streamlit dashboard for the RBC × BCCAI × SFU Beedie Agribusiness 
Analytics Hackathon. The audience is a mix of farmers, RBC lender staff, and 
technical judges. Code must be simple enough to ship in a week with 2-3 people.

### Data layer rules
- Raw CSVs live in /data/raw/ and are READ-ONLY. Never modify them.
- Derived data (joined, feature-engineered) lives in /data/derived/ as parquet files.
- The script /build_derived.py is the single source of truth for derived data.
  Never duplicate derived-data logic elsewhere. If you need new derived columns,
  ASK before adding them, and add them to build_derived.py — not inline in notebooks.

### Naming conventions in the data
- Plot identifiers are strings like "P0001" (always uppercase P, 4-digit zero-padded)
- Greenhouse identifiers are "GH001" (always GH prefix, 3-digit zero-padded)
- Farm identifiers are "F01" (always F prefix, 2-digit zero-padded)
- Dates are pandas datetimes, never strings
- All currency is CAD; never label it "$" alone — always "CAD" in column names

### Known data quirks (don't "fix" these)
- Alert types are 99.5% "High Pest Pressure" — this is real, not a bug
- Sensor data covers 2025-02-15 to 2025-09-12; price data is 2026+ — known mismatch
- Some treatment × crop cells have n=0 or n=1 — small-N is real
- Strawberry is overrepresented (57% of plots) — real, not a sampling bug

### Analysis discipline
- Always compute a baseline before comparing to a model. Report both.
- Always show confidence intervals or sample sizes alongside claims.
- Never make a treatment claim where n < 5; flag it instead.
- Use bootstrap CIs (sample sizes are uneven), not parametric.

### Streamlit app rules
- All expensive computations use @st.cache_data
- The app reads from /data/derived/, never from /data/raw/
- Each tab is its own module in /app/tabs/, imported by main.py
- No tab depends on another tab's state — tabs share data via cached functions only

### What to ask before doing
- Adding a new dependency to pyproject.toml
- Adding new columns to any parquet in /data/derived/
- Touching /build_derived.py (it affects all three streams)
- Changing the agreed prediction target (spike_3d = stress_delta_3d_forward >= 0.10)