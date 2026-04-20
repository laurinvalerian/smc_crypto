"""
tests/test_student_brain.py — unit tests for StudentBrain hot-reload
+ schema_version validation added by Track A1.6 (2026-04-20).
"""

from __future__ import annotations

import pickle
import time

import pytest

from features.schema import SCHEMA_VERSION
from models.student_brain import StudentBrain


def _dump_pkl(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def test_student_load_rejects_wrong_schema(tmp_path):
    """_load returns None when pkl carries a schema_version != features.schema.SCHEMA_VERSION.

    Before Track A1.6 StudentBrain._load silently accepted mismatched schemas,
    producing feature-misaligned inference on prod. Mirror rl_brain_v2 guard.
    """
    bad = {"model": "stub", "schema_version": SCHEMA_VERSION + 99, "feat_names": []}
    for name in ("entry", "sl", "tp", "size"):
        _dump_pkl(tmp_path / f"student_{name}.pkl", bad)

    brain = StudentBrain({"student_brain": {"enabled": True, "models_dir": str(tmp_path)}})
    # All heads were rejected → brain cannot be enabled.
    assert brain._entry_model is None
    assert brain._sl_model is None
    assert brain._tp_model is None
    assert brain._size_model is None
    assert brain.enabled is False


def test_student_reload_detects_mtime_change(tmp_path):
    """check_and_reload_models picks up updated pkl files via mtime polling.

    Pre-Track-A1.6 Student had no reload path; models could only refresh on
    a full service restart. This test pins the hot-reload contract so a
    future refactor that re-introduces a cold-start requirement fails.
    """
    good_v1 = {"model": "v1", "schema_version": SCHEMA_VERSION, "feat_names": []}
    for name in ("entry", "sl", "tp", "size"):
        _dump_pkl(tmp_path / f"student_{name}.pkl", good_v1)

    brain = StudentBrain({"student_brain": {"enabled": True, "models_dir": str(tmp_path)}})
    assert brain.enabled is True
    assert brain._entry_model is not None
    assert brain._entry_model["model"] == "v1"

    # Filesystem mtime resolution on HFS+/APFS/ext4 ≈ 1s; sleep so the new
    # write produces a distinguishable mtime, otherwise the poll is a no-op.
    time.sleep(1.1)
    good_v2 = {"model": "v2", "schema_version": SCHEMA_VERSION, "feat_names": []}
    _dump_pkl(tmp_path / "student_entry.pkl", good_v2)

    brain.check_and_reload_models()
    assert brain._entry_model["model"] == "v2", "reload did not pick up new pkl"
    # Non-changed heads must stay on v1 — partial reload respects mtime gate.
    assert brain._sl_model["model"] == "v1"
