"""
backtest/format_ab_report.py
============================
Consolidates:
  - backtest/results/ab_test_results.json           (training + holdout eval)
  - backtest/results/shadow_replay_results.json     (live decisions re-scored)

Into a single human-readable markdown report.

Run:
    python3 -m backtest.format_ab_report
"""
from __future__ import annotations

import json
from pathlib import Path

AB_PATH = Path("backtest/results/ab_test_results.json")
REPLAY_PATH = Path("backtest/results/shadow_replay_results.json")
REPORT_PATH = Path("backtest/results/ab_report.md")


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _fmt_pct(x, decimals=0):
    if x is None:
        return "—"
    return f"{100 * x:.{decimals}f}%"


def _fmt_num(x, decimals=3):
    if x is None:
        return "—"
    try:
        return f"{x:.{decimals}f}"
    except Exception:
        return str(x)


def _fmt_signed(x, decimals=2):
    if x is None:
        return "—"
    try:
        return f"{x:+.{decimals}f}"
    except Exception:
        return str(x)


def _model_row(name: str, res: dict | None) -> str:
    if not res:
        return f"| **{name}** | — | — | — | — | — | — | — | — |"
    return (
        f"| **{name}** | "
        f"{_fmt_num(res.get('auc'), 3)} | "
        f"{_fmt_num(res.get('accuracy_live_th'), 3)} | "
        f"{_fmt_num(res.get('precision_live_th'), 3)} | "
        f"{_fmt_num(res.get('recall_live_th'), 3)} | "
        f"{res.get('accepted_live_th', 0)} | "
        f"{_fmt_signed(res.get('avg_rr_live_th'), 3)} | "
        f"{_fmt_num(res.get('pf_live_th'), 2)} | "
        f"{_fmt_pct(res.get('winrate_live_th'), 0)} |"
    )


def _agreement_table(title: str, agree: dict | None) -> str:
    lines = [f"### {title}", ""]
    if not agree:
        lines.append("_(no agreement data)_")
        return "\n".join(lines)
    lines += [
        "| Cell | Count | Wins | Losses | Sum RR | Avg RR | PF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("both_accept", "only_old_accepts", "only_new_accepts",
                "both_reject", "only_new_accept", "only_old_accept"):
        if key not in agree:
            continue
        cell = agree[key]
        lines.append(
            f"| `{key}` | {cell['count']} | {cell['wins']} | {cell['losses']} | "
            f"{_fmt_signed(cell.get('sum_rr'))} | {_fmt_signed(cell.get('avg_rr'))} | "
            f"{_fmt_num(cell.get('pf'), 2)} |"
        )
    return "\n".join(lines)


def build_report() -> str:
    ab = _load(AB_PATH)
    rp = _load(REPLAY_PATH)

    lines: list[str] = []
    lines.append("# A/B Test: Entry Filter — Old vs New")
    lines.append("")

    if ab:
        cfg = ab.get("config", {})
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- **Cutoff (fair)**: `{cfg.get('cutoff_train_end')}`")
        lines.append(f"- **Holdout**: `{cfg.get('holdout_start')}` → `{cfg.get('holdout_end')}`")
        lines.append(f"- **Val window**: `{cfg.get('val_window')}`")
        lines.append(f"- **Live threshold**: `{cfg.get('live_threshold')}`")
        lines.append(f"- **Full holdout samples**: {cfg.get('n_holdout_full')}")
        lines.append(f"- **Live-realistic samples (score ≥ 0.78)**: {cfg.get('n_holdout_live')}")
        lines.append("")

        # Splits overview
        splits = ab.get("splits", {})
        if splits:
            lines.append("## Data Splits")
            lines.append("")
            lines.append("| Split | Rows | Time range | Per class |")
            lines.append("|---|---:|---|---|")
            for name, s in splits.items():
                pc = ", ".join(f"{k}={v}" for k, v in s.get("per_class", {}).items())
                lines.append(f"| `{name}` | {s['n_rows']} | {s.get('ts_min', '—')} … {s.get('ts_max', '—')} | {pc} |")
            lines.append("")

        # Full holdout
        if "full_holdout" in ab:
            fh = ab["full_holdout"]
            lines.append("## Full Holdout Results (all SMC candidates)")
            lines.append("")
            lines.append("| Model | AUC | Acc | Prec@0.55 | Rec@0.55 | Accepted | Avg RR | PF | WR |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            lines.append(_model_row("OLD (live)", fh.get("old_model")))
            lines.append(_model_row("Model A (fair)", fh.get("model_a_fair")))
            lines.append(_model_row("Model B (prod)", fh.get("model_b_production")))
            lines.append("")
            lines.append(_agreement_table("Agreement: OLD vs Model A", fh.get("agreement_old_vs_a")))
            lines.append("")
            lines.append(_agreement_table("Agreement: OLD vs Model B", fh.get("agreement_old_vs_b")))
            lines.append("")

        # Live-realistic
        if "live_realistic_holdout" in ab:
            lr = ab["live_realistic_holdout"]
            lines.append("## Live-Realistic Holdout (alignment_score ≥ 0.78)")
            lines.append("")
            lines.append("| Model | AUC | Acc | Prec@0.55 | Rec@0.55 | Accepted | Avg RR | PF | WR |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            lines.append(_model_row("OLD (live)", lr.get("old_model")))
            lines.append(_model_row("Model A (fair)", lr.get("model_a_fair")))
            lines.append(_model_row("Model B (prod)", lr.get("model_b_production")))
            lines.append("")
            lines.append(_agreement_table("Agreement: OLD vs Model A", lr.get("agreement_old_vs_a")))
            lines.append("")
            lines.append(_agreement_table("Agreement: OLD vs Model B", lr.get("agreement_old_vs_b")))
            lines.append("")

    if rp:
        lines.append("## Shadow Replay (real live decisions)")
        lines.append("")
        cfg = rp.get("config", {})
        lines.append(f"- **Log file**: `{cfg.get('log_path')}`")
        lines.append(f"- **Window**: `{cfg.get('window')}`")
        lines.append(f"- **Live threshold**: `{cfg.get('live_threshold')}`")
        lines.append(f"- **Total matched decisions**: {rp.get('summary', {}).get('n_matched', 0)}")
        lines.append("")
        summary = rp.get("summary", {})
        if summary:
            lines.append("| Source | Accepted | Accept% | Wins | Losses | Sum RR | PF | WR |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for name, label in [("old_live", "Live XGB (old)"),
                                 ("model_a", "Model A (fair)"),
                                 ("model_b", "Model B (prod)")]:
                s = summary.get(name)
                if not s:
                    lines.append(f"| {label} | — | — | — | — | — | — | — |")
                    continue
                lines.append(
                    f"| {label} | {s['accepted']} | {_fmt_pct(s.get('accept_rate'), 0)} | "
                    f"{s['wins']} | {s['losses']} | {_fmt_signed(s.get('sum_rr'))} | "
                    f"{_fmt_num(s.get('pf'), 2)} | {_fmt_pct(s.get('winrate'), 0)} |"
                )
            lines.append("")

        lines.append(_agreement_table("Agreement: live vs Model A", rp.get("agreement_old_vs_a")))
        lines.append("")
        lines.append(_agreement_table("Agreement: live vs Model B", rp.get("agreement_old_vs_b")))
        lines.append("")

        per_sym = rp.get("per_symbol", {})
        if per_sym:
            lines.append("### Per-symbol shadow replay")
            lines.append("")
            lines.append("| Symbol | N | Live conf (mean) | Model A conf | Model B conf | Winrate | Sum RR |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for sym, s in sorted(per_sym.items(), key=lambda x: -x[1]["n"]):
                lines.append(
                    f"| {sym} | {s['n']} | {_fmt_num(s.get('conf_live_mean'), 3)} | "
                    f"{_fmt_num(s.get('conf_a_mean'), 3)} | {_fmt_num(s.get('conf_b_mean'), 3)} | "
                    f"{_fmt_pct(s.get('winrate'), 0)} | {_fmt_signed(s.get('sum_rr'))} |"
                )
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"Report written to {REPORT_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    main()
