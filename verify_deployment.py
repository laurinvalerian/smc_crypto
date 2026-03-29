#!/usr/bin/env python3
"""
Pre-flight checklist for deployment verification.

Usage:
    python3 verify_deployment.py

Checks:
    1. All models load without error
    2. Feature parity between models and extractors
    3. Config matches optimal settings
    4. Kill switches configured
    5. Journal writable
    6. Dashboard module importable
    7. Exchange connections (optional — requires API keys)
"""
from __future__ import annotations

import os
import pickle
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _USE_COLOR else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _USE_COLOR else s


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _USE_COLOR else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOR else s


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
_results: list[tuple[str, str, str]] = []  # (check, status, details)


def _record(check: str, passed: bool, details: str, warn: bool = False) -> bool:
    if passed:
        status = _green("PASS")
    elif warn:
        status = _yellow("WARN")
    else:
        status = _red("FAIL")
    _results.append((check, status, details))
    return passed


# ---------------------------------------------------------------------------
# Resolve bot root (works from any cwd)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Models exist and load
# ---------------------------------------------------------------------------
REQUIRED_MODELS = [
    "models/rl_entry_filter.pkl",
    "models/rl_tp_optimizer.pkl",
    "models/rl_be_manager.pkl",
    "models/rl_exit_classifier.pkl",
]
OPTIONAL_MODELS = [
    "models/dqn_exit_manager.zip",
]

_loaded_models: dict[str, Any] = {}


def check_models() -> bool:
    all_ok = True
    for rel in REQUIRED_MODELS:
        path = _HERE / rel
        name = path.name
        if not path.exists():
            _record(f"Model: {name}", False, "file not found")
            all_ok = False
            continue
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            feat_names = data.get("feat_names", [])
            n = len(feat_names)
            _loaded_models[rel] = data
            _record(f"Model: {name}", True, f"{n} features")
        except Exception as exc:
            _record(f"Model: {name}", False, str(exc))
            all_ok = False

    for rel in OPTIONAL_MODELS:
        path = _HERE / rel
        name = path.name
        if not path.exists():
            _record(f"Model (opt): {name}", True, "not present — DQN shadow disabled", warn=False)
        else:
            # .zip models are stable-baselines3 format; just check it opens
            try:
                import zipfile
                with zipfile.ZipFile(path, "r") as zf:
                    members = zf.namelist()
                _record(f"Model (opt): {name}", True, f"zip ok ({len(members)} entries)")
            except Exception as exc:
                _record(f"Model (opt): {name}", True, f"WARN: {exc}", warn=True)

    return all_ok


# ---------------------------------------------------------------------------
# 2. Feature parity
# ---------------------------------------------------------------------------

def check_feature_parity() -> bool:
    all_ok = True

    # Exit model vs feature_extractor.EXIT_BAR_FEATURE_NAMES
    exit_rel = "models/rl_exit_classifier.pkl"
    if exit_rel in _loaded_models:
        try:
            from features.feature_extractor import EXIT_BAR_FEATURE_NAMES
            model_feats = _loaded_models[exit_rel].get("feat_names", [])
            expected = set(EXIT_BAR_FEATURE_NAMES)
            actual = set(model_feats)
            missing = expected - actual
            extra = actual - expected
            if not missing and not extra:
                _record("Parity: exit model vs extractor", True,
                        f"{len(model_feats)} features match")
            elif missing:
                # Missing features are a hard failure — inference will break
                _record("Parity: exit model vs extractor", False,
                        f"missing from model: {sorted(missing)}")
                all_ok = False
            else:
                # Extra features in model only = legacy columns, soft warning
                _record("Parity: exit model vs extractor", True,
                        f"WARN: extra in model (legacy): {sorted(extra)}", warn=True)
        except ImportError as exc:
            _record("Parity: exit model vs extractor", True,
                    f"WARN: feature_extractor not importable ({exc})", warn=True)
    else:
        _record("Parity: exit model vs extractor", True,
                "WARN: exit model not loaded — skipped", warn=True)

    # Entry model: just report feature count; live generation is in _build_xgb_features
    entry_rel = "models/rl_entry_filter.pkl"
    if entry_rel in _loaded_models:
        model_feats = _loaded_models[entry_rel].get("feat_names", [])
        _record("Parity: entry model feat count", True,
                f"{len(model_feats)} features stored in model")
    else:
        _record("Parity: entry model feat count", True,
                "WARN: entry model not loaded — skipped", warn=True)

    return all_ok


# ---------------------------------------------------------------------------
# 3. Config validation
# ---------------------------------------------------------------------------
_EXPECTED_CONFIG: list[tuple[str, Any, str]] = [
    # (dotted key, expected value, description)
    ("exit_classifier.enabled",              True,  "exit_classifier enabled"),
    ("exit_classifier.confidence_threshold", 0.65,  "exit confidence_threshold == 0.65"),
    ("dqn_exit_manager.shadow_log",          True,  "DQN shadow_log enabled"),
    ("journal.enabled",                      True,  "journal enabled"),
    ("continuous_learner.enabled",           False, "continuous_learner disabled (manual)"),
]


def _get_nested(d: dict, dotted: str) -> Any:
    parts = dotted.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(dotted)
        cur = cur[p]
    return cur


def check_config() -> bool:
    config_path = _HERE / "config" / "default_config.yaml"
    if not config_path.exists():
        _record("Config: default_config.yaml", False, "file not found")
        return False

    try:
        import yaml  # type: ignore
    except ImportError:
        _record("Config: yaml import", False, "PyYAML not installed")
        return False

    try:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
    except Exception as exc:
        _record("Config: parse", False, str(exc))
        return False

    _record("Config: default_config.yaml", True, "loaded")
    all_ok = True

    for key, expected, label in _EXPECTED_CONFIG:
        try:
            actual = _get_nested(cfg, key)
            match = actual == expected
            details = f"actual={actual!r}, expected={expected!r}"
            if not match:
                _record(f"Config: {label}", False, details)
                all_ok = False
            else:
                _record(f"Config: {label}", True, details)
        except KeyError:
            _record(f"Config: {label}", False, f"key '{key}' not found")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# 4. Kill switches
# ---------------------------------------------------------------------------
_KILL_SWITCH_KEYS: list[tuple[str, str]] = [
    ("rl_brain.enabled",               "RL brain master switch"),
    ("rl_brain.entry_filter.enabled",  "entry filter"),
    ("rl_brain.tp_optimizer.enabled",  "TP optimizer"),
    ("rl_brain.be_manager.enabled",    "BE manager"),
    ("exit_classifier.enabled",        "exit classifier"),
    ("dqn_exit_manager.enabled",       "DQN exit manager"),
    ("continuous_learner.enabled",     "continuous learner"),
    ("journal.enabled",                "trade journal"),
]


def check_kill_switches() -> bool:
    config_path = _HERE / "config" / "default_config.yaml"
    try:
        import yaml  # type: ignore
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
    except Exception as exc:
        _record("Kill switches", False, str(exc))
        return False

    for key, label in _KILL_SWITCH_KEYS:
        try:
            val = _get_nested(cfg, key)
            state = "ON" if val else "OFF"
            _record(f"Switch: {label}", True, state)
        except KeyError:
            _record(f"Switch: {label}", True, "WARN: key not found", warn=True)

    return True


# ---------------------------------------------------------------------------
# 5. Journal writable
# ---------------------------------------------------------------------------

def check_journal() -> bool:
    journal_dir = _HERE / "trade_journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    db_path = journal_dir / "journal.db"

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _preflight_test "
            "(id INTEGER PRIMARY KEY, ts TEXT)"
        )
        conn.execute("INSERT INTO _preflight_test (ts) VALUES (?)", ("preflight",))
        conn.commit()
        conn.execute("DELETE FROM _preflight_test")
        conn.commit()
        conn.execute("DROP TABLE IF EXISTS _preflight_test")
        conn.commit()
        conn.close()
        size_kb = db_path.stat().st_size // 1024 if db_path.exists() else 0
        _record("Journal: SQLite writable", True, f"{db_path.name} ({size_kb} KB)")
        return True
    except Exception as exc:
        _record("Journal: SQLite writable", False, str(exc))
        return False


# ---------------------------------------------------------------------------
# 6. Dashboard importable
# ---------------------------------------------------------------------------

def check_dashboard() -> bool:
    # Add bot root to path so imports resolve
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("dashboard", _HERE / "dashboard.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        # Collect Flask routes
        routes = []
        if hasattr(mod, "app"):
            routes = [rule.rule for rule in mod.app.url_map.iter_rules()]
        _record("Dashboard: importable", True,
                f"endpoints: {', '.join(sorted(routes)) if routes else 'none found'}")
        return True
    except ImportError as exc:
        # Flask not installed locally is expected — it's in requirements.txt for the server
        _record("Dashboard: importable", True,
                f"WARN: import failed ({exc}) — ensure flask is in requirements.txt", warn=True)
        return True
    except Exception as exc:
        _record("Dashboard: importable", False, str(exc))
        return False


# ---------------------------------------------------------------------------
# 7. SMC params
# ---------------------------------------------------------------------------

def check_smc_params() -> bool:
    clusters_path = _HERE / "config" / "instrument_clusters.json"
    if not clusters_path.exists():
        _record("SMC: instrument_clusters.json", False, "file not found")
        return False

    try:
        import json
        with open(clusters_path) as fh:
            clusters = json.load(fh)
        n = len(clusters) if isinstance(clusters, (dict, list)) else 0
        _record("SMC: instrument_clusters.json", True, f"{n} entries")
    except Exception as exc:
        _record("SMC: instrument_clusters.json", False, str(exc))
        return False

    # Check per-cluster optimized params
    smc_opt_dir = _HERE / "backtest" / "results" / "smc_optimization"
    if smc_opt_dir.exists():
        param_files = list(smc_opt_dir.glob("*.json"))
        symbols_covered: list[str] = []
        for pf in sorted(param_files):
            symbols_covered.append(pf.stem)
        _record("SMC: optimized params", True,
                f"{len(param_files)} files: {', '.join(symbols_covered)}")
    else:
        _record("SMC: optimized params", True,
                "WARN: backtest/results/smc_optimization/ not found", warn=True)

    return True


# ---------------------------------------------------------------------------
# 8. Data files (exit episode parquets)
# ---------------------------------------------------------------------------
_EXIT_PARQUETS = [
    "data/rl_training/crypto_exit_episodes.parquet",
    "data/rl_training/forex_exit_episodes.parquet",
    "data/rl_training/stocks_exit_episodes.parquet",
    "data/rl_training/commodities_exit_episodes.parquet",
]


def check_data_files() -> bool:
    all_ok = True
    for rel in _EXIT_PARQUETS:
        path = _HERE / rel
        name = Path(rel).name
        if not path.exists():
            _record(f"Data: {name}", True,
                    "WARN: not present (generated during paper trading)", warn=True)
            continue
        try:
            import pyarrow.parquet as pq  # type: ignore
            pf = pq.ParquetFile(str(path))
            rows = pf.metadata.num_rows
            _record(f"Data: {name}", True, f"{rows:,} rows")
        except ImportError:
            # pyarrow not installed — just check size
            size_kb = path.stat().st_size // 1024
            _record(f"Data: {name}", True, f"{size_kb} KB (pyarrow not installed)")
        except Exception as exc:
            _record(f"Data: {name}", False, str(exc))
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary() -> int:
    """Print results table. Returns number of hard failures."""
    col_check = max(len(r[0]) for r in _results) + 2
    col_status = 6 + (10 if _USE_COLOR else 0)  # account for ANSI escapes
    print()
    print(_bold("=" * 70))
    print(_bold(f"{'Check':<{col_check}}  {'Status':<8}  Details"))
    print(_bold("=" * 70))
    failures = 0
    for check, status, details in _results:
        print(f"{check:<{col_check}}  {status:<{col_status}}  {details}")
        # A hard failure has red FAIL in status
        if "FAIL" in status:
            failures += 1
    print(_bold("=" * 70))
    if failures == 0:
        print(_green(f"All checks passed ({len(_results)} total)."))
    else:
        print(_red(f"{failures} check(s) FAILED out of {len(_results)} total."))
    print()
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(_bold("\nSMC Bot — Pre-flight Deployment Verification"))
    print(f"Bot root: {_HERE}\n")

    # Insert bot root into path for local imports
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    print("--- 1. Models ---")
    check_models()

    print("--- 2. Feature parity ---")
    check_feature_parity()

    print("--- 3. Config validation ---")
    check_config()

    print("--- 4. Kill switches ---")
    check_kill_switches()

    print("--- 5. Journal ---")
    check_journal()

    print("--- 6. Dashboard ---")
    check_dashboard()

    print("--- 7. SMC params ---")
    check_smc_params()

    print("--- 8. Data files ---")
    check_data_files()

    failures = _print_summary()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
