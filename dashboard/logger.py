"""
Trade Logger — JSONL + SQLite logging with calibration tracking.
"""

import json
import os
import sqlite3
import time
import logging
from dataclasses import asdict
from typing import Optional

from core.market import TradeRecord

logger = logging.getLogger(__name__)

DB_PATH = "trades.db"


class TradeLogger:
    """Logs trades and tracks calibration metrics."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id TEXT,
                side TEXT,
                category TEXT,
                entry_price REAL,
                exit_price REAL,
                size_usd REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                true_prob REAL,
                net_edge_at_entry REAL,
                sources_used TEXT,
                entry_time REAL,
                exit_time REAL,
                exit_reason TEXT,
                market_question TEXT,
                actual_outcome INTEGER DEFAULT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                markets_scanned INTEGER,
                opportunities_found INTEGER,
                trades_executed INTEGER,
                categories TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS source_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                category TEXT,
                estimated_prob REAL,
                actual_outcome INTEGER,
                timestamp REAL
            )
        """)
        conn.commit()
        conn.close()

    def log_trade(self, record: TradeRecord):
        """Log a completed trade to SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO trades (
                condition_id, side, category, entry_price, exit_price,
                size_usd, pnl_usd, pnl_pct, true_prob, net_edge_at_entry,
                sources_used, entry_time, exit_time, exit_reason, market_question
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.condition_id, record.side, record.category,
                record.entry_price, record.exit_price,
                record.size_usd, record.pnl_usd, record.pnl_pct,
                record.true_prob, record.net_edge_at_entry,
                json.dumps(record.sources_used),
                record.entry_time, record.exit_time,
                record.exit_reason, record.market_question,
            ),
        )
        conn.commit()
        conn.close()

    def log_scan(self, markets: int, opportunities: int, trades: int, categories: dict):
        """Log a scan cycle."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO scans (timestamp, markets_scanned, opportunities_found, trades_executed, categories) VALUES (?, ?, ?, ?, ?)",
            (time.time(), markets, opportunities, trades, json.dumps(categories)),
        )
        conn.commit()
        conn.close()

    def log_source_accuracy(self, source: str, category: str, prob: float, outcome: int):
        """Log a source's estimate vs actual outcome for Brier scoring."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO source_accuracy (source, category, estimated_prob, actual_outcome, timestamp) VALUES (?, ?, ?, ?, ?)",
            (source, category, prob, outcome, time.time()),
        )
        conn.commit()
        conn.close()

    def get_stats(self) -> dict:
        """Get overall trading statistics."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Overall stats
        row = conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl_usd) as total_pnl,
                AVG(pnl_usd) as avg_pnl,
                AVG(CASE WHEN pnl_usd > 0 THEN pnl_usd END) as avg_win,
                AVG(CASE WHEN pnl_usd <= 0 THEN pnl_usd END) as avg_loss,
                MIN(pnl_usd) as worst_trade,
                MAX(pnl_usd) as best_trade
            FROM trades
        """).fetchone()

        stats = dict(row) if row else {}

        # Per-category stats
        cats = conn.execute("""
            SELECT category,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl_usd) as pnl,
                AVG(net_edge_at_entry) as avg_edge
            FROM trades
            GROUP BY category
        """).fetchall()
        stats["by_category"] = [dict(c) for c in cats]

        # Per-source Brier scores
        brier = conn.execute("""
            SELECT source, category,
                AVG((estimated_prob - actual_outcome) * (estimated_prob - actual_outcome)) as brier_score,
                COUNT(*) as samples
            FROM source_accuracy
            GROUP BY source, category
        """).fetchall()
        stats["source_brier"] = [dict(b) for b in brier]

        # Recent trades
        recent = conn.execute("""
            SELECT * FROM trades ORDER BY exit_time DESC LIMIT 10
        """).fetchall()
        stats["recent_trades"] = [dict(r) for r in recent]

        conn.close()
        return stats
