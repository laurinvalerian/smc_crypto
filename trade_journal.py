"""
===================================================================
 trade_journal.py  —  Trade Lifecycle Logger
 -----------------------------------------------------------
 Records the full lifecycle of every paper/live trade to SQLite
 for use as training data for the sequential exit classifier.

 Three tables:
   • trades          — one row per closed trade (metadata + outcome)
   • trade_bars      — one row per confirmed 5m bar during trade
   • post_trade_bars — 50 bars after exit (opportunity cost analysis)

 Usage in live_multi_bot.py:
   journal = TradeJournal()                       # at Runner init
   journal.open_trade(trade_id, symbol, ...)       # on entry fill
   journal.record_bar(trade_id, bar_index, ...)    # on each 5m close
   journal.close_trade(trade_id, ...)              # in _record_close()
   journal.record_post_trade_bars(trade_id, bars)  # async, 50 bars after
   journal.export_to_parquet("data/rl_training")   # for ML training
===================================================================
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id          TEXT PRIMARY KEY,
    symbol            TEXT NOT NULL,
    asset_class       TEXT NOT NULL,
    direction         TEXT NOT NULL,
    style             TEXT,
    tier              TEXT,
    entry_time        TEXT,
    exit_time         TEXT,
    entry_price       REAL,
    sl_original       REAL,
    tp                REAL,
    exit_price        REAL,
    bars_held         INTEGER,
    leverage          INTEGER,
    outcome           TEXT,
    exit_reason       TEXT,
    pnl_pct           REAL,
    rr_target         REAL,
    rr_actual         REAL,
    risk_pct          REAL,
    score             REAL,
    xgb_confidence    REAL,
    be_triggered      INTEGER DEFAULT 0,
    max_favorable_pct REAL,
    max_adverse_pct   REAL,
    post_missed_pct   REAL,
    entry_features    TEXT
)
"""

_CREATE_TRADE_BARS = """
CREATE TABLE IF NOT EXISTS trade_bars (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id         TEXT NOT NULL,
    bar_index        INTEGER NOT NULL,
    timestamp        TEXT NOT NULL,
    close            REAL,
    high             REAL,
    low              REAL,
    volume           REAL,
    unrealized_pnl_pct REAL,
    sl_distance_pct  REAL,
    rsi_5m           REAL,
    adx_1h           REAL,
    structure_break  INTEGER DEFAULT 0,
    new_fvg_against  INTEGER DEFAULT 0,
    new_ob_against   INTEGER DEFAULT 0,
    max_favorable_seen REAL,
    UNIQUE(trade_id, bar_index)
)
"""

_CREATE_POST_BARS = """
CREATE TABLE IF NOT EXISTS post_trade_bars (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id       TEXT NOT NULL,
    bar_index      INTEGER NOT NULL,
    timestamp      TEXT,
    high           REAL,
    low            REAL,
    close          REAL,
    cumulative_pct REAL,
    UNIQUE(trade_id, bar_index)
)
"""

_CREATE_REJECTED_SIGNALS = """
CREATE TABLE IF NOT EXISTS rejected_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    asset_class     TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     REAL,
    sl_price        REAL,
    tp_price        REAL,
    xgb_confidence  REAL,
    alignment_score REAL,
    entry_features  TEXT,
    -- Scalp horizon: 72 bars (6h) — would it have been a good scalp?
    outcome_scalp   TEXT,
    mfe_scalp       REAL,
    mae_scalp       REAL,
    -- Day horizon: 288 bars (24h) — would it have been a good day trade?
    outcome_day     TEXT,
    mfe_day         REAL,
    mae_day         REAL,
    -- Swing horizon: 1440 bars (5d) — would it have been a good swing?
    outcome_swing   TEXT,
    mfe_swing       REAL,
    mae_swing       REAL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_trade_bars_trade ON trade_bars(trade_id)",
    "CREATE INDEX IF NOT EXISTS idx_post_bars_trade  ON post_trade_bars(trade_id)",
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_trades_class     ON trades(asset_class)",
    "CREATE INDEX IF NOT EXISTS idx_rejected_symbol  ON rejected_signals(symbol)",
]


class TradeJournal:
    """
    Persistent trade lifecycle logger backed by SQLite (WAL mode).

    Thread-safety: All writes are synchronous and must be called from
    the same thread/event-loop.  In live_multi_bot.py's single asyncio
    loop this is guaranteed — every on_candle / _record_close call
    runs sequentially without preemption between awaits.
    """

    def __init__(self, db_path: str = "trade_journal/journal.db") -> None:
        path = Path(db_path)
        if db_path != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(_CREATE_TRADES)
        self._conn.execute(_CREATE_TRADE_BARS)
        self._conn.execute(_CREATE_POST_BARS)
        self._conn.execute(_CREATE_REJECTED_SIGNALS)
        for idx_sql in _CREATE_INDEXES:
            self._conn.execute(idx_sql)
        self._conn.commit()

        # Track running max_favorable per open trade (in-memory, avoids reads)
        self._max_favorable: dict[str, float] = {}

        logger.info("TradeJournal initialised: %s", db_path)

    # ------------------------------------------------------------------ #
    #  Write API                                                           #
    # ------------------------------------------------------------------ #

    def open_trade(
        self,
        trade_id: str,
        symbol: str,
        asset_class: str,
        direction: str,
        style: str,
        entry_time: datetime,
        entry_price: float,
        *,
        tier: str = "",
        sl_original: float,
        tp: float,
        score: float,
        rr_target: float,
        leverage: int,
        risk_pct: float,
        entry_features: dict[str, float] | None = None,
        xgb_confidence: float = 0.0,
    ) -> None:
        """Record trade open.  Call after bracket order is confirmed filled."""
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO trades
                   (trade_id, symbol, asset_class, direction, style, tier,
                    entry_time, entry_price, sl_original, tp, score,
                    xgb_confidence, rr_target, leverage, risk_pct, entry_features)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade_id, symbol, asset_class, direction, style, tier,
                    _ts(entry_time), entry_price, sl_original, tp, score,
                    xgb_confidence, rr_target, leverage, risk_pct,
                    json.dumps(entry_features or {}),
                ),
            )
            self._conn.commit()
            self._max_favorable[trade_id] = 0.0
        except Exception as exc:
            logger.error("journal.open_trade failed for %s: %s", trade_id, exc)

    def record_bar(
        self,
        trade_id: str,
        bar_index: int,
        timestamp: datetime,
        close: float,
        high: float,
        low: float,
        volume: float,
        unrealized_pnl_pct: float,
        sl_distance_pct: float,
        rsi_5m: float = 0.0,
        adx_1h: float = 0.0,
        structure_break: bool = False,
        new_fvg_against: bool = False,
        new_ob_against: bool = False,
    ) -> None:
        """Record one closed 5m bar while trade is active.
        Must be called only on confirmed closed candles (not poll ticks)."""
        try:
            # Track running max favorable
            current_max = self._max_favorable.get(trade_id, 0.0)
            favorable = max(unrealized_pnl_pct, 0.0)
            if favorable > current_max:
                current_max = favorable
                self._max_favorable[trade_id] = current_max

            self._conn.execute(
                """INSERT OR IGNORE INTO trade_bars
                   (trade_id, bar_index, timestamp, close, high, low, volume,
                    unrealized_pnl_pct, sl_distance_pct, rsi_5m, adx_1h,
                    structure_break, new_fvg_against, new_ob_against,
                    max_favorable_seen)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade_id, bar_index, _ts(timestamp),
                    close, high, low, volume,
                    unrealized_pnl_pct, sl_distance_pct,
                    rsi_5m, adx_1h,
                    int(structure_break), int(new_fvg_against), int(new_ob_against),
                    current_max,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.error("journal.record_bar failed for %s bar %d: %s",
                         trade_id, bar_index, exc)

    def close_trade(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        outcome: str,
        exit_reason: str,
        bars_held: int,
        pnl_pct: float,
        rr_actual: float,
        max_favorable_pct: float,
        max_adverse_pct: float,
        be_triggered: bool = False,
    ) -> None:
        """Record trade close.  Call inside _record_close()."""
        try:
            self._conn.execute(
                """UPDATE trades SET
                   exit_time=?, exit_price=?, outcome=?, exit_reason=?,
                   bars_held=?, pnl_pct=?, rr_actual=?,
                   max_favorable_pct=?, max_adverse_pct=?,
                   be_triggered=?
                   WHERE trade_id=?""",
                (
                    _ts(exit_time), exit_price, outcome, exit_reason,
                    bars_held, pnl_pct, rr_actual,
                    max_favorable_pct, max_adverse_pct,
                    int(be_triggered),
                    trade_id,
                ),
            )
            self._conn.commit()
            self._max_favorable.pop(trade_id, None)
        except Exception as exc:
            logger.error("journal.close_trade failed for %s: %s", trade_id, exc)

    def record_rejected_signal(
        self,
        symbol: str,
        asset_class: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        xgb_confidence: float,
        alignment_score: float,
        entry_features: dict[str, float] | None = None,
    ) -> int | None:
        """Record a signal rejected by the XGBoost entry filter.

        Returns the row ID (for outcome tracking) or None on failure.
        """
        try:
            cursor = self._conn.execute(
                """INSERT INTO rejected_signals
                   (timestamp, symbol, asset_class, direction, entry_price,
                    sl_price, tp_price, xgb_confidence, alignment_score,
                    entry_features)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol, asset_class, direction, entry_price,
                    sl_price, tp_price, xgb_confidence, alignment_score,
                    json.dumps(entry_features or {}),
                ),
            )
            self._conn.commit()
            return cursor.lastrowid
        except Exception as exc:
            logger.error("journal.record_rejected_signal failed for %s: %s", symbol, exc)
            return None

    def update_rejection_outcome(
        self,
        rejection_id: int,
        horizon: str,
        outcome: str,
        mfe: float,
        mae: float,
    ) -> None:
        """Write counterfactual outcome for a rejected signal.

        horizon: "scalp" (72 bars/6h), "day" (288 bars/24h), "swing" (1440 bars/5d)
        outcome: "win" (TP would hit), "loss" (SL would hit), "timeout"
        mfe/mae: max favorable/adverse excursion as fraction of entry price
        """
        col_map = {
            "scalp": ("outcome_scalp", "mfe_scalp", "mae_scalp"),
            "day": ("outcome_day", "mfe_day", "mae_day"),
            "swing": ("outcome_swing", "mfe_swing", "mae_swing"),
        }
        cols = col_map.get(horizon)
        if not cols:
            return
        out_col, mfe_col, mae_col = cols
        try:
            self._conn.execute(
                f"UPDATE rejected_signals SET {out_col}=?, {mfe_col}=?, {mae_col}=? "
                f"WHERE id=?",
                (outcome, mfe, mae, rejection_id),
            )
            self._conn.commit()
        except Exception as exc:
            logger.error("journal.update_rejection_outcome failed id=%d: %s",
                         rejection_id, exc)

    def record_post_trade_bars(
        self,
        trade_id: str,
        bars: list[dict[str, Any]],
        exit_price: float,
        direction: str,
    ) -> None:
        """
        Record up to 50 bars after exit for opportunity cost analysis.

        bars: list of OHLCV dicts with keys: timestamp, high, low, close
        exit_price: price at which trade was closed
        direction: "long" or "short"
        """
        if not bars:
            return
        try:
            rows = []
            for i, bar in enumerate(bars[:50]):
                bar_close = float(bar.get("close", 0.0))
                if direction == "long":
                    cumulative_pct = (bar_close - exit_price) / exit_price if exit_price > 0 else 0.0
                else:
                    cumulative_pct = (exit_price - bar_close) / exit_price if exit_price > 0 else 0.0
                rows.append((
                    trade_id, i,
                    str(bar.get("timestamp", "")),
                    float(bar.get("high", 0.0)),
                    float(bar.get("low", 0.0)),
                    bar_close,
                    cumulative_pct,
                ))

            self._conn.executemany(
                """INSERT OR IGNORE INTO post_trade_bars
                   (trade_id, bar_index, timestamp, high, low, close, cumulative_pct)
                   VALUES (?,?,?,?,?,?,?)""",
                rows,
            )

            # Compute post_missed_pct: max cumulative move in trade direction
            # over the 50 post-trade bars (positive = left money on table)
            if rows:
                post_pcts = [r[6] for r in rows]
                post_missed = max(post_pcts) if post_pcts else 0.0
                self._conn.execute(
                    "UPDATE trades SET post_missed_pct=? WHERE trade_id=?",
                    (post_missed, trade_id),
                )

            self._conn.commit()
        except Exception as exc:
            logger.error("journal.record_post_trade_bars failed for %s: %s",
                         trade_id, exc)

    # ------------------------------------------------------------------ #
    #  Read API (for tests and training data export)                      #
    # ------------------------------------------------------------------ #

    def get_trade(self, trade_id: str) -> dict[str, Any] | None:
        """Return trade row as dict, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM trades WHERE trade_id=?", (trade_id,)
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.execute(
            "SELECT * FROM trades WHERE trade_id=?", (trade_id,)
        ).description]
        return dict(zip(cols, row))

    def get_trade_bars(self, trade_id: str) -> list[dict[str, Any]]:
        """Return all bar rows for a trade, ordered by bar_index."""
        cur = self._conn.execute(
            "SELECT * FROM trade_bars WHERE trade_id=? ORDER BY bar_index",
            (trade_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count_closed_trades(self) -> int:
        """Return number of trades with a non-null exit_time."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL"
        ).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ #
    #  Export for ML training                                             #
    # ------------------------------------------------------------------ #

    def export_to_parquet(self, output_dir: str = "data/rl_training") -> Path:
        """
        Export closed trades + trade_bars to parquet for exit classifier training.

        Writes:
          {output_dir}/live_exit_episodes.parquet  — one row per bar per trade
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas required for export_to_parquet")
            raise

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load all closed trades
        trades_df = pd.read_sql(
            "SELECT * FROM trades WHERE exit_time IS NOT NULL",
            self._conn,
        )
        if trades_df.empty:
            logger.warning("No closed trades to export")
            return out_dir / "live_exit_episodes.parquet"

        # Load all bar data for closed trades
        closed_ids = trades_df["trade_id"].tolist()
        if not closed_ids:
            logger.warning("No closed trade IDs to export")
            return out_dir / "live_exit_episodes.parquet"

        placeholders = ",".join(["?" for _ in closed_ids])
        bars_df = pd.read_sql(
            f"SELECT * FROM trade_bars WHERE trade_id IN ({placeholders})",
            self._conn,
            params=closed_ids,
        )
        if bars_df.empty:
            logger.warning("No trade bars to export")
            return out_dir / "live_exit_episodes.parquet"

        # Merge trade metadata into bars
        trade_meta = trades_df[[
            "trade_id", "symbol", "asset_class", "direction", "style", "tier",
            "outcome", "pnl_pct", "rr_actual", "rr_target", "score",
            "leverage", "risk_pct", "bars_held",
        ]]
        merged = bars_df.merge(trade_meta, on="trade_id", how="left")

        # Compute label_hold_better (two-pass, post-hoc) without losing trade_id.
        # At each bar, does continuing to hold improve the final outcome?
        # remaining_rr = final_rr - rr_at_this_bar
        # label = 1 if remaining_rr > 0 (hold was better), 0 if exit was better
        risk_pct_map = trade_meta.set_index("trade_id")["risk_pct"].to_dict()
        rr_actual_map = trade_meta.set_index("trade_id")["rr_actual"].to_dict()
        outcome_map = trade_meta.set_index("trade_id")["outcome"].to_dict()

        merged = merged.copy()
        bar_risk = merged["trade_id"].map(risk_pct_map).clip(lower=1e-6)
        bar_rr_actual = merged["trade_id"].map(rr_actual_map).fillna(0.0)
        # Sign: rr_actual is positive for wins, negative for losses
        bar_unrealized_rr = merged["unrealized_pnl_pct"] / bar_risk
        remaining_rr = bar_rr_actual - bar_unrealized_rr
        merged["bar_unrealized_rr"] = bar_unrealized_rr
        merged["label_hold_better"] = (remaining_rr > 0).astype(int)

        out_path = out_dir / "live_exit_episodes.parquet"
        merged.to_parquet(out_path, index=False)
        logger.info(
            "Exported %d bar rows from %d trades → %s",
            len(merged), merged["trade_id"].nunique(), out_path,
        )
        return out_path

    def close(self) -> None:
        """Close the SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _ts(dt: datetime | None) -> str | None:
    """Serialize datetime to ISO string, always UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()
