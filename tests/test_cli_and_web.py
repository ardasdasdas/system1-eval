import json
import pathlib
import threading
import urllib.request
from system1_eval.cli import main
from system1_eval.web import create_app_server


def test_cli_validate_run_and_report(tmp_path: pathlib.Path, capsys):
    dataset_dir = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    out_path = tmp_path / "leaderboard.json"

    rc_val = main(["validate", "--dataset-dir", str(dataset_dir)])
    assert rc_val == 0

    rc_run = main(["run", "--dataset-dir", str(dataset_dir), "--output", str(out_path)])
    assert rc_run == 0
    assert out_path.exists()

    rc_rep = main(["report", "--results", str(out_path)])
    assert rc_rep == 0
    captured = capsys.readouterr().out
    assert "kev-4b-v1" in captured
    assert "clef-flash" in captured


def test_web_leaderboard_endpoints_and_ui(tmp_path: pathlib.Path):
    dataset_dir = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    out_path = tmp_path / "leaderboard.json"
    main(["run", "--dataset-dir", str(dataset_dir), "--output", str(out_path)])

    server = create_app_server("127.0.0.1", 0, dataset_dir=dataset_dir, results_path=out_path)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            html = resp.read().decode("utf-8")
            assert "System1-Eval" in html
            assert "Reliability Diagram" in html

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/leaderboard") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data["models"]) >= 6

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/scenarios") as resp:
            scenarios = json.loads(resp.read().decode("utf-8"))
            assert len(scenarios["scenarios"]) >= 30
    finally:
        server.shutdown()
        server.server_close()
