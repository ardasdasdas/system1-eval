import pathlib
from system1_eval.adapters import build_systemone_request_payload, parse_systemone_response
from system1_eval.runner import format_terminal_report, run_benchmark
from system1_eval.schema import load_dataset_dir


def test_systemone_payload_and_response_parser():
    dataset_dir = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    scenarios = load_dataset_dir(dataset_dir)
    choice_sc = next(s for s in scenarios if s.question["type"] == "choice")

    payload = build_systemone_request_payload("typesafe/jev-latest", choice_sc)
    assert payload["model"] == "typesafe/jev-latest"
    assert payload["state"] == choice_sc.state
    qid = choice_sc.question["id"]
    assert payload["questions"][qid]["type"] == "choice"

    fake_resp = {
        "model": "typesafe/jev-latest",
        "answers": {
            qid: {
                "type": "choice",
                "choice": choice_sc.ground_truth["answer"],
                "confidence": 0.91,
            }
        },
        "usage": {"input_tokens": 95, "output_tokens": 0},
    }
    pred = parse_systemone_response(choice_sc, "typesafe/jev-latest", fake_resp, latency_ms=14.2)
    assert pred.predicted_answer == choice_sc.ground_truth["answer"]
    assert pred.confidence == 0.91
    assert pred.output_tokens == 0


def test_run_benchmark_and_terminal_report(tmp_path: pathlib.Path):
    dataset_dir = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    out_file = tmp_path / "leaderboard.json"
    result = run_benchmark(dataset_dir=dataset_dir, output_path=out_file)
    assert out_file.exists()
    assert len(result["models"]) >= 6

    kev = next(m for m in result["models"] if m["model_id"] == "kev-4b-v1")
    llama = next(m for m in result["models"] if m["model_id"] == "llama-3.1-8b-vllm")
    # System One models have zero output tokens, lower latency, and better calibration (lower ECE) than uncalibrated LLM JSON
    assert kev["overall"]["avg_output_tokens"] == 0
    assert kev["overall"]["ece"] < llama["overall"]["ece"]
    assert kev["overall"]["p50_ms"] < llama["overall"]["p50_ms"]

    report = format_terminal_report(result)
    assert "kev-4b-v1" in report
    assert "ECE" in report
