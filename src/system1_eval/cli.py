from __future__ import annotations

import argparse
import json
import pathlib
import sys

from system1_eval.runner import format_terminal_report, run_benchmark
from system1_eval.schema import VALID_TRACKS, DatasetValidationError, load_dataset_dir
from system1_eval.web import create_app_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="system1-eval",
        description="System1-Eval (DecisionArena): Benchmark & Calibration Platform for System One Decision Models (Clef, Kev, Jev, Tev1)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    val_p = subparsers.add_parser("validate", help="Validate crowdsourced JSONL dataset files")
    val_p.add_argument("--dataset-dir", default="datasets", help="Path to datasets directory")

    run_p = subparsers.add_parser("run", help="Run evaluation harness and compute calibration/latency metrics")
    run_p.add_argument("--dataset-dir", default="datasets", help="Path to datasets directory")
    run_p.add_argument("--output", default="results/leaderboard.json", help="Output leaderboard JSON path")
    run_p.add_argument("--models", default="all", help="Comma-separated model IDs or 'all'")

    rep_p = subparsers.add_parser("report", help="Print formatted terminal leaderboard table")
    rep_p.add_argument("--results", default="results/leaderboard.json", help="Path to leaderboard JSON file")

    srv_p = subparsers.add_parser("serve", help="Launch live dynamic leaderboard and calibration web UI")
    srv_p.add_argument("--host", default="127.0.0.1", help="Bind host (use 0.0.0.0 for containers/HF Spaces)")
    srv_p.add_argument("--port", type=int, default=7860, help="Bind port (default 7860)")
    srv_p.add_argument("--dataset-dir", default="datasets", help="Path to datasets directory")
    srv_p.add_argument("--results", default="results/leaderboard.json", help="Path to leaderboard JSON file")

    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            scenarios = load_dataset_dir(args.dataset_dir)
        except DatasetValidationError as exc:
            print(f"[ERROR] Dataset validation failed: {exc}", file=sys.stderr)
            return 1
        by_track = {t: sum(1 for s in scenarios if s.track == t) for t in VALID_TRACKS}
        print(f"[OK] Validated {len(scenarios)} scenarios across {len(VALID_TRACKS)} tracks: {by_track}")
        return 0

    if args.command == "run":
        model_ids = None if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
        result = run_benchmark(dataset_dir=args.dataset_dir, output_path=args.output, model_ids=model_ids)
        print(format_terminal_report(result))
        return 0

    if args.command == "report":
        res_path = pathlib.Path(args.results)
        if not res_path.exists():
            print(f"[ERROR] Results file not found: {res_path}", file=sys.stderr)
            return 1
        data = json.loads(res_path.read_text(encoding="utf-8"))
        print(format_terminal_report(data))
        return 0

    if args.command == "serve":
        server = create_app_server(
            host=args.host,
            port=args.port,
            dataset_dir=args.dataset_dir,
            results_path=args.results,
        )
        actual_port = server.server_address[1]
        print(f"System1-Eval Leaderboard running at http://{args.host}:{actual_port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
