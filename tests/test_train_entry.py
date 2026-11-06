import json
import sys

import train


def test_rtx3060_profile_preserves_the_champion_architecture_and_limits_workers():
    config = json.loads(train.GPU_CONFIG.read_text(encoding="utf-8"))
    assert config["device"] == "cuda"
    assert (config["channels"], config["blocks"], config["se"]) == (128, 10, True)
    assert config["gumbel"] is True
    assert config["selfplay_workers"] == 1
    assert config["concurrent_games"] <= config["games"]
    assert config["batch"] <= 128
    assert config["replay_on_disk"] is True


def test_default_training_output_is_in_the_repository_runs_directory():
    assert train.DEFAULT_OUTPUT == train.ROOT / "runs" / "rtx3060_6gb"


def test_dry_run_selects_gen85_without_starting_training(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(train, "detect", lambda: ("cuda", "test RTX 3060, 6.0 GB VRAM"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--dry-run", "--output", str(tmp_path / "run")],
    )
    assert train.main() == 0
    output = capsys.readouterr().out
    assert "gen85_best.pt" in output
    assert "Dry run" in output


def test_custom_config_does_not_implicitly_load_the_champion(monkeypatch, capsys, tmp_path):
    config = tmp_path / "custom.json"
    config.write_text(
        json.dumps({"channels": 8, "blocks": 1, "batch": 4, "games": 1, "concurrent_games": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(train, "detect", lambda: ("cuda", "test GPU"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--dry-run", "--config", str(config)])
    assert train.main() == 0
    assert "Initial weights:" not in capsys.readouterr().out
