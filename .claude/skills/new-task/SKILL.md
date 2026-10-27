---
name: new-task
description: Turn a rough description, or an item from the plan, into one or more well-formed GitHub issues an agent can finish without asking questions. Use when anyone wants to create, split or re-scope tasks, write an experiment issue, or file a bug.
argument-hint: "[what needs doing]"
---

# Write a task issue

A good issue lets an agent finish the work alone, without understanding the whole system.

1. **Ground it.** Read the relevant parts of `docs/PRD.md`, `docs/ARCHITECTURE.md`, `docs/API-CONTRACT.md`, `docs/PROTOCOL.md`, the plan stage it comes from (`specs/2026-10-27-the-plan.md`), and the actual files involved.
2. **Size it.** About 1 to 2 hours of work, about 300 changed lines or fewer, one module. Split anything bigger and note the order.
3. **Pick the type** (AGENTS.md, "Work types"): `code`, `probe`, `experiment`, `analysis` or `final`. Fill the Type field and add the matching label.
   - **experiment** uses `.github/ISSUE_TEMPLATE/experiment.md`, and its Measurement is filled in full before filing: hypothesis, arms (each a method + config diff), seeds, timesteps, scenarios (validation only), primary metric, guard metrics, decision rule, budget (CPU-hours and dollars), and the A/A or baseline comparison. Never leave a field for later.
   - **final** names the exact selected checkpoints and needs Aarmen's yes before it's filed.
4. **Check the contract and protocol.** If the task needs an interface that isn't in `docs/API-CONTRACT.md`, or a protocol change, stop and propose it to Aarmen first.
5. **Check the scorer files.** If the task would change `scenarios.py`, `rewards.py`, `envs.py`, `metrics.py` or `docs/PROTOCOL.md`, it gets the `core` label, and its Steps say what the diff should be, small enough for Aarmen to review in one sitting.
6. **Fill in the template** (`.github/ISSUE_TEMPLATE/task.md` or `experiment.md`): exact file paths; concrete steps that name the functions and contract interfaces; Done-when items that are runnable checks (a pytest command, a CLI output, a ledger row); Out of scope naming the tempting extras.
7. **Show the draft** to the user. On approval, `gh issue create --title "..." --body-file <tmp> --label task --label <type>` (`--label bug` instead of `task` for a defect; add `--label core` when step 5 applies).
8. Report the issue URL(s).
