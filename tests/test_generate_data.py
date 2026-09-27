"""Data generation preserves declared sizes and records honest provenance."""
import hashlib
import json
from pathlib import Path

import pytest

from nsga2_sats.generate_data import DEFAULT_SCALES, generate_dataset


def read_rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def test_paper_scales_counts_and_valid_windows(tmp_path):
    manifest = generate_dataset(tmp_path / "large", seed=321)
    assert [(item["tasks"], item["antennas"]) for item in manifest["instances"]] == list(DEFAULT_SCALES)
    for item in manifest["instances"]:
        antenna_names = set()
        requests = set()
        for file_info in item["files"]:
            path = tmp_path / "large" / file_info["name"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == file_info["sha256"]
            rows = read_rows(path)
            for request, antenna, start, end, duration, turnaround in rows:
                requests.add((file_info["orbit"], int(request)))
                antenna_names.add(antenna)
                start, end, duration, turnaround = map(float, (start, end, duration, turnaround))
                assert 0 <= start < end <= manifest["method"]["horizon"]
                assert 0 < duration <= end - start
                assert turnaround >= 0
        assert len(requests) == item["tasks"]
        assert len(antenna_names) == item["antennas"]
    assert manifest["data_status"] == "generated-paper-scale"
    assert manifest["template"] == "AFIT1"


def test_seed_repeatability_is_directory_independent(tmp_path):
    left, right, other = (tmp_path / name for name in ("left", "right", "other"))
    first = generate_dataset(left, seed=17, scales=((40, 24),))
    repeated = generate_dataset(right, seed=17, scales=((40, 24),))
    changed = generate_dataset(other, seed=18, scales=((40, 24),))
    assert first == repeated
    assert {p.name: p.read_bytes() for p in left.iterdir()} == {p.name: p.read_bytes() for p in right.iterdir()}
    assert first["instances"][0]["files"] != changed["instances"][0]["files"]
    assert str(tmp_path) not in (left / "generation-manifest.json").read_text()


def test_existing_outputs_are_preserved(tmp_path):
    target = tmp_path / "protected"
    target.mkdir()
    existing = target / "result.csv"
    existing.write_text("original result\n")
    with pytest.raises(FileExistsError):
        generate_dataset(target)
    assert existing.read_text() == "original result\n"
    assert list(target.iterdir()) == [existing]


@pytest.mark.parametrize("scales", [(), ((10, 24),), ((40, 2),)])
def test_invalid_shapes_do_not_create_outputs(tmp_path, scales):
    with pytest.raises(ValueError):
        generate_dataset(tmp_path / "invalid", scales=scales)
    assert not (tmp_path / "invalid").exists()
