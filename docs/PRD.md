# GreenLeaf 2.0: research brief

## The pitch
At the RBC × BCCAI × SFU hackathon, GreenLeaf warned growers before their plants got stressed, and placed 2nd.
GreenLeaf 2.0 acts on those warnings. It is an AI co-pilot that runs a tomato greenhouse on top of the grower's own
rules. It looks for extra profit and is never allowed to break the plants' climate limits more often than the grower
does.

Full pitch page: https://claude.ai/artifact/SpQ5Dxb5dU8Pt513s3HNJk

## The claim we're testing
On weather it never trained on (Amsterdam, London and Abbotsford BC, 2012), a constrained residual RL controller on a
Bayesian-optimized grower controller keeps climate violations at or below the grower's, at higher profit. Every number
comes with 95% CIs over seeds × scenarios.

We report the result whichever way it comes out.

## Research questions
1. **Reproduction:** does PPO beat upstream's rule-based controller on the paper's own season, as GreenLight-Gym
   reports (E2)?
2. **The baseline:** how much of RL's reported advantage disappears when the rule-based controller is tuned with the
   same effort (E1 vs E3)?
3. **The contribution:** at budgets of 100%, 50% and 25% of the tuned controller's violations, what profit does the
   constrained residual reach, and does it keep the budget (E5, confirmed in E6)?
4. **Generalization:** do these hold on a new year (Amsterdam 2012) and new climates (London, Abbotsford BC)? This is
   the `final`, run once.

## Who it's for
- **Hiring managers for AI/ML engineer roles:** it shows RL done carefully, with a reproduction, a fair baseline,
  constraints, held-out evaluation and honest statistics.
- **Anyone curious:** the demo shows the grower and the AI side by side through a season.

## Non-goals
- **No claim to beat real growers or Autonomous Greenhouse Challenge teams.** GreenLight models climate control, not
  strategy such as plant density or harvest timing.
- **No yield calibration to real data** (a research project of its own). A climate-only calibration to AGC data is a
  stretch, capped at 2–3 days.
- **No LLM in the core.** An optional "Ask GreenLeaf" explainer agent comes last, in its own plan.

## The demo story (built in its own plan, after the final)
1. **Season replay:** pick a location and press play. The grower on the left, the AI on the right; climate inside the
   safe band; actuators lighting up; a profit counter; an end-of-season scoreboard.
2. **Live run:** any city, year and start date, plus a safety-budget slider. Both controllers simulate a season in
   seconds on a free CPU.
3. **The trade-off chart:** profit against violation-hours across the three budgets.
4. **Optional:** "Ask GreenLeaf" explains a decision with a what-if re-simulation.
