#!/usr/bin/env python3
"""Command-line interface: run | baseline | gate."""

import argparse
import sys
from pathlib import Path

import yaml

from .gate import gate as run_gate
from .gate import save_baseline
from .report import write_reports
from .runner import run_suites
from .suite import load_suites


def _load_config(path):
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _add_common(ap):
    ap.add_argument("--suites", default="suites", help="suite file or directory")
    ap.add_argument("--config", default="lmeval.config.yaml")
    ap.add_argument("--models", nargs="*", help="override models (provider:model ...)")
    ap.add_argument("--only", nargs="*", help="limit to named suites")
    ap.add_argument("--deterministic-only", action="store_true",
                    help="skip llm_judge graders (use in CI without judge access)")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="stop the run once cumulative USD cost reaches this budget")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="run up to N tasks in parallel (default 1 = sequential)")


def _text_table(rows):
    if not rows:
        return "(no results)"
    return "\n".join(
        f"{r['suite']:<18} {r['model']:<26} "
        f"pass={r['pass_rate']} ci=[{r['pass_rate_lo']},{r['pass_rate_hi']}] "
        f"judge={r['mean_judge']} "
        f"cost=${r['cost_usd']} p50={r['p50_latency_s']}s p95={r['p95_latency_s']}s"
        for r in rows
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lmeval", description="LLM eval harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run suites and write a report")
    _add_common(p_run)
    p_run.add_argument("--out", default="results")

    p_base = sub.add_parser("baseline", help="run suites and save a baseline")
    _add_common(p_base)
    p_base.add_argument("--name", default="default")
    p_base.add_argument("--baseline-dir", default="baselines")

    p_gate = sub.add_parser("gate", help="run suites and fail on regression")
    _add_common(p_gate)
    p_gate.add_argument("--name", default="default")
    p_gate.add_argument("--baseline-dir", default="baselines")
    p_gate.add_argument("--tolerance", type=float, default=0.0)
    p_gate.add_argument("--min-pass-rate", type=float, default=None)
    p_gate.add_argument("--out", default="results")

    args = parser.parse_args(argv)
    config = _load_config(args.config)
    suites = load_suites(args.suites, only=args.only)
    if not suites:
        print("no suites found")
        return 1

    results = run_suites(
        suites, config,
        cli_models=args.models,
        deterministic_only=getattr(args, "deterministic_only", False),
        max_cost=args.max_cost,
        workers=args.concurrency,
    )

    if args.cmd == "run":
        paths = write_reports(results, args.out)
        print("\n" + _text_table(paths["rows"]))
        print("\nwrote:", {k: paths[k] for k in ("json", "csv", "md", "transcripts")})
        return 0

    if args.cmd == "baseline":
        path = Path(args.baseline_dir) / f"{args.name}.json"
        save_baseline(results, path)
        print(f"saved baseline -> {path}")
        return 0

    if args.cmd == "gate":
        write_reports(results, args.out)
        path = Path(args.baseline_dir) / f"{args.name}.json"
        ok, lines = run_gate(results, path,
                             tolerance=args.tolerance,
                             min_pass_rate=args.min_pass_rate)
        print("\n".join(lines))
        print("\nGATE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
