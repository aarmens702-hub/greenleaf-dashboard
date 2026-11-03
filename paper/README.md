# Workshop paper draft (issue #66)

An anonymized draft for the AAAI-27 workshop "Physics-Governed AI for Energy" (PGAI4E), in-progress track: 3 to 4 pages, double-blind, non-archival. Working title: "Tune the Baseline, Check the Objective: A Pre-Registered Evaluation of Residual RL for Greenhouse Climate Control". It condenses `docs/REPORT.md` into an evaluation paper. Nothing in it was trained or scored for the paper: every number is copied from recorded files.

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper, built from `AnonymousSubmission2027.tex` in the author kit |
| `refs.bib` | The verified reference list, unchanged; all 31 entries are cited |
| `figures/make_figure.py` | Draws Figure 1 from the three test `summary.csv` files and prints every value it plots |
| `figures/fig_main.pdf` | Figure 1, written by `make_figure.py` (vector, 3.3 in wide) |
| `.gitignore` | Ignores the pdfLaTeX build artifacts and `main.pdf` |
| `aaai2027.sty`, `aaai2027.bst` | The AAAI-27 author kit's style files, as distributed, unmodified |

## Author kit

AAAI-27 author kit, `\pdfinfo{/TemplateVersion (2027.1)}`, with `aaai2027.sty` identifying itself as `2027/10/21 AAAI 2027 Submission format`. The style files are byte-identical to the kit:

```
01990f8d3716f8e39c5b28a7478f3ac6e28630d1161de6105a7217f5e7f60e43  aaai2027.sty
5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c  aaai2027.bst
```

`main.tex` loads `aaai2027` with the `submission` option, which prints "Anonymous submission" and the review notice. It uses no package or command that the kit forbids (no hyperref, geometry, float or negative vspace). The only layout setting is `\setlength{\tabcolsep}{3.5pt}` in Table 1, which the kit's instructions allow.

## Build

`aaai2027.sty` requires pdfLaTeX: its engine guard (`\RequirePDFTeX`) stops XeLaTeX and LuaLaTeX, and so stops tectonic, which runs XeTeX. Never disable the guard in this folder. Build with pdfLaTeX and BibTeX from this directory:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or: latexmk -pdf main.tex   (Overleaf also works: upload this folder)
```

On the Mac this draft was written on, pdfLaTeX comes from TinyTeX (TeX Live 2026, pdfTeX 1.40.29), user-installed with no sudo:

```bash
export PATH=$HOME/Library/TinyTeX/bin/universal-darwin:$PATH
tlmgr install newtx xpatch xstring carlisle fontaxes placeins caption tex-gyre   # beyond TinyTeX's base set
```

Build in a scratch copy of this folder so the artifacts stay out of the repo (`.gitignore` covers them if you build here). The pdfLaTeX build gives 6 pages with no overfull boxes: everything before the references, including the AI-use and reproducibility statements, ends on the last line of page 4's right column with no line to spare (both columns of page 4 are full), and the references fill page 5 and the top of page 6. Any added line in the body will push the reproducibility statement onto page 5, and so will a taller Figure 1 (at 4.4 in it costs two lines); re-check after every edit.

Before submitting, clear the PDF's metadata (the kit asks for this), and strip the LaTeX comments before posting the source anywhere public: they name the study's code labels and repository paths.

## Where each number comes from

Every number in the text, Table 1 and the captions has a LaTeX comment, next to it or just above it, naming its source:

- `<split>/summary.csv <label> <column>` means `control/results/analysis/<split>/summary.csv`, row `<label>`, column `<column>`. The test splits are `amsterdam_2012`, `london_2012` and `abbotsford_2012` (final-result-aee23915); validation is `validation`, whose labels carry their issue (for example `residual_w05 #36/f2fd`).
- Each Table 1 row has one comment that names the file, the label, `n_runs`, `invalid_episodes`, and the columns its cells come from, in order.
- `REPORT.md Sec. n` means `docs/REPORT.md` at the commit this draft was written on. It is used for setup facts (season length, controls, ranges, prices, hyperparameters, splits) and for counts that REPORT.md made from the final's files and that no summary.csv holds: the seed-0 replay hours and the CO₂-range seasons (29 and 18 of 240 for the residual agents; 15 and 11 of 120 for `ppo` and `ppo_paper`). Those counts were copied from REPORT.md §5.2, not recounted for the paper.
- Derived percentages show their arithmetic in the comment. They are ratios of IQMs, rounded to whole percent, with no CI:
  - the share of PPO's lead over the default rules that tuning recovers is `-g(rb_default) / (-g(rb_default) + g(ppo))`, where `g` is `profit_gain_iqm` (73%, 76%, 74%; 74% on validation);
  - PPO's fruit-growth shortfall is `1 - ppo / rb_tuned` on `yield_kg_m2_iqm` (31%, 28%, 32%);
  - the residual agent's hour reduction is `1 - residual_w05 / rb_tuned` on `viol_hours_sum_iqm` (20%, 25%, 21%; Amsterdam's 19.9% rounds to 20%).
- Rounding is Python's `round()` to the shown precision on the recorded values (round-half-even on the stored binary value, as `docs/REPORT.md` §3.7): 2 decimals for €/m² and kg/m², 1 for hours. For example, the stored 551.45 h (Abbotsford's `lag_b100` Σ) is 551.4500000000000455 in binary and rounds to 551.5 h.

The related-work facts about SafeGreenRL and Bie et al. come from the verified reference notes for this issue, not from REPORT.md. The introduction's description of GreenLight-Gym's baseline (fixed grower setpoints, untuned; PPO outperforms it; its authors note that tuning the rules would give a more competitive baseline) is paraphrased from arXiv 2410.05336v2, the version of the cited IFAC paper, as checked for this issue. Their comments in `main.tex` say so.

## The figure

`figures/fig_main.pdf` is drawn by `figures/make_figure.py` from `control/results/analysis/{amsterdam,london,abbotsford}_2012/summary.csv`. Regenerate it with control's environment:

```bash
cd control && uv run python ../paper/figures/make_figure.py
```

The script prints every value it plots. Panel (a) splits PPO's profit lead over the default rules into the tuned rules' gain over the default rules (`-profit_gain_iqm` of `rb_default`) and PPO's gain over the tuned rules (`profit_gain_iqm` of `ppo`), labelled with tuning's share. Panel (b) plots `profit_eur_m2_iqm` against `yield_kg_m2_iqm` for six controllers on all three splits: marker shape is the controller, fill is the city. The city key uses filled rectangles, not markers, since every marker shape names a controller. The figure is 3.3 in wide, the column width, so it is included at scale and all its text, legends included, prints at 9 pt (the kit asks for at least 9 pt in illustrations). No legend covers a data point. It uses display names only, has no source footer, and embeds its fonts as TrueType (`pdf.fonttype` 42), not Type 3.

## References

`refs.bib` is the verified list. Each entry was checked on 2026-11-02/03 against arXiv, publisher, PMLR, JMLR, NeurIPS proceedings or Crossref records. No entry's facts were added or edited. The paper's only em dashes are inside the titles of the two Hemming et al. references (`---` in `refs.bib`), kept verbatim. The paper attributes nothing to a reference beyond what the issue's brief, REPORT.md and the check of arXiv 2410.05336v2 (GreenLight-Gym, above) state.
