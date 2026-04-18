"""SHAP + Permutation-Importance auf die 13 Alignment-Komponenten.

Input:  data/rl_training/crypto_samples.parquet (9.4M rows, windows 0-12)
Target: label_profitable (binary) and optimal_entry (Teacher v2 hindsight)
Output: .omc/research/alignment-ablation.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

PARQUET = Path("data/rl_training/crypto_samples.parquet")
OUT_MD = Path(".omc/research/alignment-ablation.md")
OUT_MD.parent.mkdir(parents=True, exist_ok=True)

CORE_FEATURES = [
    "daily_bias",          # 0.12 (encoded as bullish=1/bearish=-1/neutral=0 in training)
    "bias_strong",         # 0.08
    "h4_confirms",         # 0.08
    "h4_poi",              # 0.08
    "h1_confirms",         # 0.08
    "h1_choch",            # 0.06
    "has_entry_zone",      # 0.15
    "precision_trigger",   # 0.15
    "volume_ok",           # 0.10
]
BONUS_PROXIES = [
    "adx_1h",              # 0.02 — proxy for adx_strong
    "hour_sin",            # 0.02 — proxy for session_optimal (1/2)
    "hour_cos",            # 0.02 — proxy for session_optimal (2/2)
    "rsi_1h",              # 0.02 — proxy for momentum_confluent (rsi-side)
    "atr_1h_norm",         # 0.02 — proxy for momentum (vol-side)
    "decay_1h",            # 0.02 — proxy for zone_freshness (1h level)
    "decay_15m",           #      — freshness close to entry
    "struct_1h",           #      — structural agreement proxy for tf_agreement
    "struct_4h",
    "struct_1d",
]

FEATURES = CORE_FEATURES + BONUS_PROXIES
TARGETS = ["label_profitable", "optimal_entry"]

WINDOW_TRAIN = list(range(0, 10))   # 0-9 = ~7.7M rows
WINDOW_TEST = [10, 11]              # OOS ~1.55M rows

SUBSAMPLE_PER_CLASS = 100_000       # balanced-ish subsample for training


def load_subsample(target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with features + target, balanced subsample for train."""
    cols = FEATURES + [target, "window", "symbol"]
    df = pd.read_parquet(PARQUET, columns=cols)

    # Encode daily_bias if string
    if df["daily_bias"].dtype == "object" or str(df["daily_bias"].dtype) == "string":
        df["daily_bias"] = df["daily_bias"].map({"bullish": 1, "bearish": -1}).fillna(0).astype("float32")

    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    train = df[df["window"].isin(WINDOW_TRAIN)]
    test = df[df["window"].isin(WINDOW_TEST)]

    # Balanced subsample for training (keeps positives, subsamples negatives)
    pos = train[train[target] == 1]
    neg = train[train[target] == 0]
    n_pos = min(SUBSAMPLE_PER_CLASS, len(pos))
    n_neg = min(SUBSAMPLE_PER_CLASS, len(neg))
    train_sub = pd.concat([
        pos.sample(n=n_pos, random_state=42),
        neg.sample(n=n_neg, random_state=42),
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Test: 100K stratified sample (keep natural class ratio)
    test_sub = test.sample(n=min(100_000, len(test)), random_state=42).reset_index(drop=True)

    return train_sub, test_sub


def train_xgb(X_train, y_train, X_test, y_test) -> xgb.XGBClassifier:
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def run_analysis(target: str) -> dict:
    print(f"\n═══ Target: {target} ═══")
    t0 = time.time()
    train_df, test_df = load_subsample(target)
    print(f"  loaded: train={len(train_df):,} (pos={int((train_df[target]==1).sum()):,}), test={len(test_df):,} (pos={int((test_df[target]==1).sum()):,})  [{time.time()-t0:.1f}s]")

    X_train = train_df[FEATURES].astype("float32").values
    y_train = train_df[target].values
    X_test = test_df[FEATURES].astype("float32").values
    y_test = test_df[target].values

    t0 = time.time()
    model = train_xgb(X_train, y_train, X_test, y_test)
    preds = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, preds))
    print(f"  trained: AUC={auc:.3f}  [{time.time()-t0:.1f}s]")

    # Permutation importance on test set (fast with n_repeats=3)
    t0 = time.time()
    # Subsample test for perm-imp speed (10K is plenty)
    idx = np.random.RandomState(42).choice(len(X_test), size=min(10_000, len(X_test)), replace=False)
    perm = permutation_importance(
        model, X_test[idx], y_test[idx], n_repeats=5, random_state=42, n_jobs=-1, scoring="roc_auc",
    )
    perm_ranking = sorted(
        zip(FEATURES, perm.importances_mean, perm.importances_std),
        key=lambda x: -x[1],
    )
    print(f"  perm-imp: done  [{time.time()-t0:.1f}s]")

    # SHAP values via TreeExplainer (exact, fast)
    t0 = time.time()
    shap_sample = X_test[idx]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(shap_sample)
    # TreeExplainer on XGBClassifier returns single array for binary
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_ranking = sorted(
        zip(FEATURES, mean_abs_shap),
        key=lambda x: -x[1],
    )
    print(f"  shap: done  [{time.time()-t0:.1f}s]")

    # Gain importance (from model itself)
    gain = model.get_booster().get_score(importance_type="gain")
    # Map f0/f1 → feature names
    gain_map = {f"f{i}": name for i, name in enumerate(FEATURES)}
    gain_named = {gain_map[k]: v for k, v in gain.items() if k in gain_map}
    for f in FEATURES:
        gain_named.setdefault(f, 0.0)
    gain_ranking = sorted(gain_named.items(), key=lambda x: -x[1])

    return {
        "target": target,
        "auc": auc,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "pos_rate_test": float((y_test == 1).mean()),
        "perm": [
            {"feature": f, "mean": float(m), "std": float(s)} for f, m, s in perm_ranking
        ],
        "shap": [
            {"feature": f, "mean_abs": float(v)} for f, v in shap_ranking
        ],
        "gain": [
            {"feature": f, "gain": float(g)} for f, g in gain_ranking
        ],
    }


def render_markdown(results: list[dict]) -> str:
    lines = ["# Alignment-Score Komponenten — SHAP + Permutation-Importance", ""]
    lines.append(f"**Datum**: 2026-04-18")
    lines.append(f"**Datenquelle**: `{PARQUET}` (9.38M rows, windows 0-12)")
    lines.append(f"**Train-Split**: windows {WINDOW_TRAIN[0]}-{WINDOW_TRAIN[-1]} (balanced subsample {SUBSAMPLE_PER_CLASS:,}/class)")
    lines.append(f"**Test-Split**: windows {WINDOW_TEST} (OOS, 100K natural-ratio sample)")
    lines.append(f"**Features**: {len(FEATURES)} = 9 core (training weights) + {len(BONUS_PROXIES)} bonus proxies")
    lines.append("")
    lines.append(f"## Feature-Set (14 total)")
    lines.append(f"- **Core 9** (training weights): {', '.join(CORE_FEATURES)}")
    lines.append(f"- **Bonus proxies**: {', '.join(BONUS_PROXIES)}")
    lines.append("")

    for res in results:
        lines.append(f"## Target: `{res['target']}`")
        lines.append(f"- OOS AUC: **{res['auc']:.3f}**")
        lines.append(f"- N train: {res['n_train']:,} | N test: {res['n_test']:,} | pos-rate test: {res['pos_rate_test']:.2%}")
        lines.append("")
        lines.append("### Top-10 by Permutation-Importance (AUC drop when feature shuffled)")
        lines.append("| Rank | Feature | Mean ΔAUC | Std | Core/Bonus |")
        lines.append("|------|---------|-----------|-----|------------|")
        for i, row in enumerate(res["perm"][:14]):
            tag = "Core" if row["feature"] in CORE_FEATURES else "Bonus"
            lines.append(f"| {i+1} | `{row['feature']}` | {row['mean']:+.4f} | {row['std']:.4f} | {tag} |")
        lines.append("")
        lines.append("### Top-10 by SHAP (mean |value|)")
        lines.append("| Rank | Feature | mean \\|SHAP\\| | Core/Bonus |")
        lines.append("|------|---------|--------------|------------|")
        for i, row in enumerate(res["shap"][:14]):
            tag = "Core" if row["feature"] in CORE_FEATURES else "Bonus"
            lines.append(f"| {i+1} | `{row['feature']}` | {row['mean_abs']:.4f} | {tag} |")
        lines.append("")
        lines.append("### Bottom-5 (Kandidaten für Entfernung)")
        # Union of bottom-5 of perm and shap
        perm_bottom = [r["feature"] for r in res["perm"][-5:]]
        shap_bottom = [r["feature"] for r in res["shap"][-5:]]
        union = set(perm_bottom) | set(shap_bottom)
        intersect = set(perm_bottom) & set(shap_bottom)
        lines.append(f"- Perm-bottom-5: {', '.join(f'`{f}`' for f in perm_bottom)}")
        lines.append(f"- SHAP-bottom-5: {', '.join(f'`{f}`' for f in shap_bottom)}")
        lines.append(f"- **Schnittmenge (sicher unwichtig)**: {', '.join(f'`{f}`' for f in intersect) or '_leer_'}")
        lines.append(f"- **Vereinigung (zu hinterfragen)**: {', '.join(f'`{f}`' for f in union)}")
        lines.append("")

    lines.append("## Interpretation & Empfehlung für Phase 2.1 (SSOT-Extraktion)")
    lines.append("")
    lines.append("- Features die in BEIDEN Targets UND BEIDEN Methoden (perm+shap) in den Bottom-5 landen: entfernen.")
    lines.append("- Bonus-Komponenten (nur Live, nicht Training): wenn unwichtig, aus `_multi_tf_alignment_score` streichen.")
    lines.append("- Core-Komponenten mit signifikantem Beitrag: SSOT-Implementation in `core/alignment.py` behalten.")
    lines.append("")
    lines.append("**Hinweis**: Bonus-Proxies (adx_1h, rsi_1h, hour_sin/cos, atr_1h_norm, decay_*, struct_*) sind _numerische Approximationen_ der tatsächlichen Live-Flags. Echte Bonus-Flags (adx_strong, session_optimal, momentum_confluent, tf_agreement, zone_fresh) existieren nicht im Parquet, weil sie nur in `_multi_tf_alignment_score` live berechnet werden. Ihre Importance hier ist indikativ, nicht exakt.")
    return "\n".join(lines)


def main() -> None:
    results = []
    for target in TARGETS:
        res = run_analysis(target)
        results.append(res)

    md = render_markdown(results)
    OUT_MD.write_text(md)
    # Also write raw JSON for downstream
    (OUT_MD.with_suffix(".json")).write_text(json.dumps(results, indent=2))
    print(f"\nWritten: {OUT_MD}")
    print(f"Written: {OUT_MD.with_suffix('.json')}")


if __name__ == "__main__":
    main()
