import pathlib
from system1_eval.schema import VALID_TRACKS, load_dataset_dir


def test_crowdsourced_datasets_load_and_cover_all_tracks():
    root = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    scenarios = load_dataset_dir(root)
    assert len(scenarios) >= 30
    by_track = {t: [s for s in scenarios if s.track == t] for t in VALID_TRACKS}
    for track, items in by_track.items():
        assert len(items) >= 10, f"Track {track} has fewer than 10 scenarios"
