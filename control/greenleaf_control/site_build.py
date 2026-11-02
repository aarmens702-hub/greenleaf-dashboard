"""Assemble the deployable demo site from recorded files only (#58, S1), one command:

    python -m greenleaf_control.site_build --out DIR [--force]

copies site/'s plain files and site/data/*.json (its validation replays) into DIR at the same
relative paths, the final's recorded 2012 replays of the page's controllers into DIR/data/final/
under their own names, byte for byte, and writes DIR/data/index.json (what the page can play) and
DIR/data/results.json (each split's summary.csv numbers, copied, never recomputed). The final's
directory comes from the ledger's final-result row, never typed. It plays no season, reads no
weather, loads no env and writes nothing outside DIR.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from greenleaf_control import ledger, manifest

# The page's controllers, id -> display name, in its order. random is left out: 3 invalid
# Abbotsford seasons, and noise is not a controller.
CONTROLLERS = {
    "rb_tuned": "Tuned rules",
    "rb_default": "Default rules",
    "residual_w05": "Residual agent, fixed penalty",
    "lag_b100": "Residual agent, constrained",
    "ppo": "PPO, no rules",
    "ppo_paper": "PPO, paper setup",
    "zero_controls": "Do nothing",
}
# (split id, location, year, label, dir under out), in index.json's order.
SPLITS = (
    ("amsterdam_2012", "Amsterdam", 2012, "Amsterdam 2012", "data/final"),
    ("london_2012", "London", 2012, "London 2012", "data/final"),
    ("abbotsford_2012", "Abbotsford", 2012, "Abbotsford 2012", "data/final"),
    ("validation", "Amsterdam", 2010, "Amsterdam 2010 (validation)", "data"),
)
# site/data's validation replays: controller id -> file prefix.
VALIDATION = {"rb_tuned": "rb_tuned", "residual_w05": "residual_w05_seed100"}
METRICS = (
    "profit_eur_m2",
    "profit_gain",
    "viol_hours_total",
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
    "yield_kg_m2",
)
INTS = ("n_runs", "invalid_episodes")
FLOATS = (
    *(f"{m}_{s}" for m in METRICS for s in ("iqm", "lo", "hi")),
    "viol_hours_sum_iqm",
    "rb_tuned_profit_iqm",
    "rb_tuned_viol_sum_iqm",
)
TEST_NOTE = "2012 test weather, scored once in the final ({}). Simulated, not a real greenhouse."
VALIDATION_NOTE = "Validation seasons: checkpoints were selected on these. Simulated."
# Repo-relative dirs only their own tools write; out may not be in or equal to one.
PROTECTED = ("site", "control/results", "control/runs", "control/ledger")


def _key(path: Path) -> Path:
    """path resolved (links, .., macOS's /tmp) and casefolded (APFS ignores case)."""
    return Path(str(path.resolve()).casefold())


def _check_out(out: Path, root: Path, force: bool) -> None:
    """Refuse an out inside a protected dir, the repo root or an ancestor of it, or non-empty
    without force."""
    where = _key(out)
    for rel in PROTECTED:
        if where.is_relative_to(_key(root / rel)):
            raise SystemExit(f"refused: {out} is in {rel}/, which only its own tools write")
    if _key(root).is_relative_to(where):
        raise SystemExit(f"refused: {out} is the repo root or holds it")
    if out.exists() and any(out.iterdir()) and not force:
        raise SystemExit(f"refused: {out} is not empty; --force overwrites same-named files")


def _days(src: Path, stem: str) -> list[int]:
    """The start days of the files src/<stem><day>.json present, ascending."""
    return sorted(
        int(p.name.removeprefix(stem).removesuffix(".json")) for p in src.glob(f"{stem}*.json")
    )


def _seed(src: Path, names: list[str]) -> int | None:
    """The replays' one meta.run.seed (None for a baseline); refused if they disagree."""
    seeds = set()
    for name in names:
        run = json.loads((src / name).read_bytes())["meta"]["run"]
        seeds.add(None if run is None else run["seed"])
    if len(seeds) != 1:
        raise SystemExit(f"{src}: {names[0]} and its other start days disagree on seed: {seeds}")
    return seeds.pop()


def _validation_arms(registered: dict, existing: list[dict]) -> dict[str, str]:
    """Validation's summary.csv label -> page controller id: the final's baselines by name, and each
    arm the final scored by report's label for its registration, `<arm> #<issue>/<4 hex>`."""
    arms = {b: b for b in registered["baselines"] if b in CONTROLLERS}
    for arm, runs in ledger.final_arms(registered, existing).items():
        reg_id = ledger._find(existing, next(iter(runs)), "run")["registered"]
        issue = ledger._find(existing, reg_id, "registered")["issue"]
        arms[f"{arm} #{issue}/{reg_id.removeprefix('registered-')[:4]}"] = arm
    return arms


def _rows(path: Path, arms: dict[str, str]) -> dict[str, dict]:
    """summary.csv's labels in its order but random: its numbers as float(text) and int(text), a
    blank decision null, and controller = arms.get(label)."""
    with path.open(newline="", encoding="utf-8") as f:
        found = list(csv.DictReader(f))
    return {
        r["label"]: {
            "controller": arms.get(r["label"]),
            **{c: int(r[c]) for c in INTS},
            "decision": r["decision"] or None,
            **{c: float(r[c]) for c in FLOATS},
        }
        for r in found
        if r["label"] != "random"
    }


def build(
    out: Path,
    *,
    root: Path = manifest.REPO_ROOT,
    ledger_path: Path = ledger.LEDGER,
    force: bool = False,
) -> dict:
    """Assemble the site in out; every refusal is a SystemExit before anything is made or written.
    Returns {"final_result", "replays", "files", "bytes"}: the final-result row id, the replays
    copied into out/data/final, and the count and total size of the files written."""
    finals = ledger.rows("final-result", path=ledger_path)
    if not finals:
        raise SystemExit(
            f"{ledger_path} has no final-result row: there are no recorded 2012 seasons to show"
        )
    [final] = finals
    _check_out(out, root, force)
    existing = ledger.rows(path=ledger_path)
    registered = ledger._find(existing, final["final-registered"], "final-registered")
    replays = root / "control" / "results" / "final" / registered["row_id"] / "replays"
    analysis = root / "control" / "results" / "analysis"
    site = root / "site"

    # Everything is read and checked before a dir is made or a file written.
    copies = {p.name: p for p in sorted(site.iterdir()) if p.is_file()}
    copies |= {f"data/{p.name}": p for p in sorted((site / "data").glob("*.json"))}
    index = {"final_result": final["row_id"], "splits": []}
    results = {}
    n_replays = 0
    for split, location, year, label, folder in SPLITS:
        test = split != "validation"
        src = replays if test else site / "data"
        prefixes = {c: c for c in CONTROLLERS} if test else VALIDATION
        stems = {c: f"{prefix}_{location}_{year}_" for c, prefix in prefixes.items()}
        found = {c: _days(src, stem) for c, stem in stems.items()}
        days = found[next(iter(found))]
        if not days or any(d != days for d in found.values()):
            raise SystemExit(
                f"{src}: the {label} replays aren't one same set of start days per controller:"
                f" {found}"
            )
        controllers = []
        for c, stem in stems.items():
            names = [f"{stem}{day}.json" for day in days]
            seed = _seed(src, names)
            controllers.append(
                {"id": c, "name": CONTROLLERS[c], "file_prefix": prefixes[c], "seed": seed}
            )
            if test:
                copies |= {f"{folder}/{name}": src / name for name in names}
                n_replays += len(names)
        index["splits"].append(
            {
                "id": split,
                "location": location,
                "year": year,
                "label": label,
                "dir": folder,
                "start_days": days,
                "controllers": controllers,
            }
        )
        arms = prefixes if test else _validation_arms(registered, existing)
        entry = {"final_result": final["row_id"]} if test else {}
        entry["note"] = TEST_NOTE.format(final["row_id"]) if test else VALIDATION_NOTE
        entry["rows"] = _rows(analysis / split / "summary.csv", arms)
        results[split] = entry
    texts = {
        "data/index.json": json.dumps(index, indent=2, allow_nan=False) + "\n",
        "data/results.json": json.dumps(results, indent=2, allow_nan=False) + "\n",
    }

    out.mkdir(parents=True, exist_ok=True)
    written = []
    for rel, src in copies.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out / rel)
        written.append(out / rel)
    for rel, text in texts.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text, encoding="utf-8")
        written.append(out / rel)
    return {
        "final_result": final["row_id"],
        "replays": n_replays,
        "files": len(written),
        "bytes": sum(p.stat().st_size for p in written),
    }


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    p = argparse.ArgumentParser(prog="python -m greenleaf_control.site_build")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the dir to assemble the site in; not in site/, control/results, runs or ledger",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="build into a non-empty dir, overwriting same-named files (deletes nothing)",
    )
    args = p.parse_args(argv)
    done = build(args.out, ledger_path=ledger_path, force=args.force)
    data = args.out / "data"
    print(f"copied {done['files'] - done['replays'] - 2} site files into {args.out}")
    print(f"copied {done['replays']} replays of {done['final_result']} into {data / 'final'}")
    print(f"wrote {data / 'index.json'}")
    print(f"wrote {data / 'results.json'}")
    print(f"{done['files']} files, {done['bytes']:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
