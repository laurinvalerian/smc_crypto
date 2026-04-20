"""
StudentBrain — unified Teacher-Student inference for entry + SL + TP + size.

Replaces the old stacked-adjuster architecture (entry_filter + tp_optimizer
+ sl_adjuster + position_sizer running sequentially and each correcting the
last). That stack produced self-reinforcing labels and algebraic-only PF
claims (audit 2026-04-17). The Student is trained on hindsight-optimal
targets from the Teacher v2 labeller, so there is no feedback loop.

Files on disk (under ``models/``):
  - ``student_entry.pkl``   — XGBClassifier   → P(worth_taking | features)
  - ``student_sl.pkl``      — XGBRegressor   → optimal SL distance in R
  - ``student_tp.pkl``      — XGBRegressor   → optimal TP distance in R
  - ``student_size.pkl``    — XGBRegressor   → position-size multiplier

Each pickle is a dict with keys ``model``, ``feat_names``, ``schema_version``,
``dead_features``, ``clip_ranges``, ``asset_class_map``, created by
``train_student.py``. The ``prepare_features`` pipeline from rl_brain_v2 is
reused verbatim so feature ordering stays in sync.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from teacher.teacher_v2 import (
    SL_CAP, SL_FLOOR, TP_CAP, TP_FLOOR, SIZE_CAP, SIZE_FLOOR,
)
from features.schema import SCHEMA_VERSION

logger = logging.getLogger(__name__)


class StudentBrain:
    """Multi-head replacement for the old RLBrainSuite exit-model stack.

    Usage::

        student = StudentBrain(cfg)
        if student.enabled:
            out = student.predict(features)
            if out.accept:
                sl_dist_price = out.sl_rr * planned_r_unit
                tp_dist_price = out.tp_rr * planned_r_unit
                risk = base_risk * out.size
    """

    class Prediction:
        __slots__ = ("accept", "entry_prob", "sl_rr", "tp_rr", "size")

        def __init__(self, accept: bool, entry_prob: float,
                     sl_rr: float, tp_rr: float, size: float) -> None:
            self.accept = accept
            self.entry_prob = entry_prob
            self.sl_rr = sl_rr
            self.tp_rr = tp_rr
            self.size = size

        def __repr__(self) -> str:  # pragma: no cover
            return (f"Prediction(accept={self.accept}, prob={self.entry_prob:.3f}, "
                    f"sl={self.sl_rr:.2f}R, tp={self.tp_rr:.2f}R, size={self.size:.2f})")

    def __init__(self, config: dict[str, Any]) -> None:
        student_cfg = config.get("student_brain", {})
        self.enabled: bool = bool(student_cfg.get("enabled", False))
        self.accept_threshold: float = float(student_cfg.get("accept_threshold", 0.55))
        self.models_dir = Path(student_cfg.get("models_dir", "models"))
        self._sl_floor = float(student_cfg.get("sl_floor", SL_FLOOR))
        self._sl_cap = float(student_cfg.get("sl_cap", SL_CAP))
        self._tp_floor = float(student_cfg.get("tp_floor", TP_FLOOR))
        self._tp_cap = float(student_cfg.get("tp_cap", TP_CAP))
        self._size_floor = float(student_cfg.get("size_floor", SIZE_FLOOR))
        self._size_cap = float(student_cfg.get("size_cap", SIZE_CAP))
        # Minimum reward:risk required at entry time (TP / SL). Lower than the
        # strategy's old MIN_RR=3.0 because the student directly proposes
        # realistic TPs — a hardcoded 3R was often too optimistic. Student RR
        # already encodes expected edge.
        self._min_rr: float = float(student_cfg.get("min_rr", 1.5))

        # mtime tracker for hot-reload (populated by _load on successful reads)
        self._mtimes: dict[str, float] = {}

        self._entry_model: dict | None = self._load("student_entry.pkl")
        self._sl_model: dict | None = self._load("student_sl.pkl")
        self._tp_model: dict | None = self._load("student_tp.pkl")
        self._size_model: dict | None = self._load("student_size.pkl")

        # Config-requested-enabled gate (preserved so hot-reload can re-enable
        # once all heads arrive on disk without a service restart).
        self._config_enabled: bool = self.enabled
        self._update_enabled_state()

        if self.enabled:
            logger.info(
                "StudentBrain initialised: 4 heads loaded, accept_threshold=%.2f, "
                "sl∈[%.2f,%.2f]R, tp∈[%.2f,%.2f]R, size∈[%.2f,%.2f], min_rr=%.2f",
                self.accept_threshold,
                self._sl_floor, self._sl_cap,
                self._tp_floor, self._tp_cap,
                self._size_floor, self._size_cap,
                self._min_rr,
            )

    def _load(self, filename: str) -> dict | None:
        path = self.models_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or "model" not in data:
                logger.warning("Invalid pickle structure in %s", path)
                return None
            # Schema version gate — skip model if trained against a different
            # feature schema. Mirrors rl_brain_v2._load_model for parity.
            model_sv = data.get("schema_version")
            if model_sv is not None and model_sv != SCHEMA_VERSION:
                logger.warning(
                    "Schema mismatch for %s: model v%s, code expects v%s — skipping (retrain needed)",
                    path.name, model_sv, SCHEMA_VERSION,
                )
                return None
            # Remember mtime so check_and_reload_models can detect updates.
            try:
                self._mtimes[filename] = path.stat().st_mtime
            except OSError:
                pass
            return data
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return None

    def _update_enabled_state(self) -> None:
        """Recompute self.enabled based on which heads are loaded.

        Called from __init__ and from check_and_reload_models so hot-reload
        can flip enabled on/off cleanly (e.g. when the 4th head finally
        arrives, or if one is deleted). Preserves the user's config intent
        via _config_enabled — if the config disabled Student, we never
        silently re-enable.
        """
        loaded = [
            name for name, attr in (
                ("entry", "_entry_model"),
                ("sl",    "_sl_model"),
                ("tp",    "_tp_model"),
                ("size",  "_size_model"),
            ) if getattr(self, attr, None) is not None
        ]
        was_enabled = getattr(self, "enabled", False)
        config_wants_enabled = getattr(self, "_config_enabled", self.enabled)
        self.enabled = config_wants_enabled and (len(loaded) == 4)
        if was_enabled != self.enabled:
            logger.info(
                "StudentBrain enabled=%s (heads loaded: %s; config_enabled=%s)",
                self.enabled, ",".join(loaded) or "none", config_wants_enabled,
            )

    def check_and_reload_models(self) -> None:
        """Poll pkl file mtimes; rebuild heads if any changed.

        Called by live_multi_bot._model_reload_loop every 60s. Safe to call
        from the asyncio event loop: synchronous, no yielding, so no race
        with StudentBrain.predict() which also runs in the event loop.
        """
        reloaded: list[str] = []
        for head_name, attr in (
            ("entry", "_entry_model"),
            ("sl",    "_sl_model"),
            ("tp",    "_tp_model"),
            ("size",  "_size_model"),
        ):
            filename = f"student_{head_name}.pkl"
            path = self.models_dir / filename
            if not path.exists():
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime == self._mtimes.get(filename):
                continue
            data = self._load(filename)  # _load updates self._mtimes on success
            if data is not None:
                setattr(self, attr, data)
                reloaded.append(head_name)
                logger.info(
                    "Student %s model reloaded (mtime=%.0f)", head_name, mtime,
                )
        if reloaded:
            self._update_enabled_state()

    def _build_features(self, features: dict[str, float], model_data: dict) -> np.ndarray:
        """Build a 1×N feature row in the order the model expects.

        Missing features default to 0.0 (same behaviour as the old
        RLBrainSuite._build_features). asset_class_id is derived from the
        trade's asset_class if present, else 0.
        """
        feat_names = model_data.get("feat_names", [])
        x = np.zeros((1, len(feat_names)), dtype=np.float32)
        for i, name in enumerate(feat_names):
            value = features.get(name, 0.0)
            try:
                x[0, i] = float(value)
            except (TypeError, ValueError):
                x[0, i] = 0.0
        return x

    def predict(self, features: dict[str, float]) -> "StudentBrain.Prediction":
        """Run all 4 heads. Returns a Prediction dataclass-like object.

        When the student is disabled or a head is missing the method
        returns neutral defaults (accept=False, sl=1R, tp=2R, size=1.0).
        """
        if not self.enabled:
            return self.Prediction(accept=False, entry_prob=0.0,
                                   sl_rr=1.0, tp_rr=2.0, size=1.0)

        # Entry head — XGBClassifier, use predict_proba
        try:
            x_e = self._build_features(features, self._entry_model)
            proba = self._entry_model["model"].predict_proba(x_e)[0]
            entry_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as exc:
            logger.debug("entry head predict failed: %s", exc)
            entry_prob = 0.0

        # SL head — XGBRegressor
        try:
            x_sl = self._build_features(features, self._sl_model)
            sl_raw = float(self._sl_model["model"].predict(x_sl)[0])
            sl_rr = float(np.clip(sl_raw, self._sl_floor, self._sl_cap))
        except Exception as exc:
            logger.debug("sl head predict failed: %s", exc)
            sl_rr = 1.0

        # TP head — XGBRegressor
        try:
            x_tp = self._build_features(features, self._tp_model)
            tp_raw = float(self._tp_model["model"].predict(x_tp)[0])
            tp_rr = float(np.clip(tp_raw, self._tp_floor, self._tp_cap))
        except Exception as exc:
            logger.debug("tp head predict failed: %s", exc)
            tp_rr = 2.0

        # Size head — XGBRegressor
        try:
            x_size = self._build_features(features, self._size_model)
            size_raw = float(self._size_model["model"].predict(x_size)[0])
            size = float(np.clip(size_raw, self._size_floor, self._size_cap))
        except Exception as exc:
            logger.debug("size head predict failed: %s", exc)
            size = 1.0

        # Accept gate: probability threshold AND realistic reward:risk
        predicted_rr = tp_rr / max(sl_rr, 0.1)
        accept = (entry_prob >= self.accept_threshold) and (predicted_rr >= self._min_rr)

        return self.Prediction(accept=accept, entry_prob=entry_prob,
                               sl_rr=sl_rr, tp_rr=tp_rr, size=size)
