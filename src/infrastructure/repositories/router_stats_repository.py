import sqlite3
import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

class RouterStatsRepository:
    """
    Persistence layer for model performance, costs, and effectiveness.
    Uses SQLite for transactional integrity.
    """
    def __init__(self, db_path: str = "data/router_stats.db") -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            # Table for model performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_stats (
                    request_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    timestamp DATETIME,
                    input_chars INTEGER,
                    output_chars INTEGER,
                    duration_ms REAL,
                    cost REAL,
                    effectiveness_score REAL DEFAULT 1.0,
                    topic TEXT,
                    session_id TEXT,
                    format_score REAL DEFAULT 1.0,
                    sentiment_score REAL DEFAULT 0.0,
                    judge_score REAL DEFAULT 1.0,
                    embedding_json TEXT
                )
            """)
            # Table for session context (vector similarity tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    session_id TEXT,
                    timestamp DATETIME,
                    input_text TEXT,
                    embedding_json TEXT
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_stats_model_ts ON model_stats(model_id, timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_stats_session_ts ON model_stats(session_id, timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_history_session_ts ON session_history(session_id, timestamp DESC)"
            )
            conn.commit()
        self._ensure_schema_columns()
        self._prune_old_records()

    def _ensure_schema_columns(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(model_stats)")
            rows = cursor.fetchall()
            existing_columns = {str(row[1]) for row in rows if len(row) > 1}

            if "feedback_score" not in existing_columns:
                cursor.execute("ALTER TABLE model_stats ADD COLUMN feedback_score REAL")
            if "feedback_label" not in existing_columns:
                cursor.execute("ALTER TABLE model_stats ADD COLUMN feedback_label TEXT")
            if "feedback_comment" not in existing_columns:
                cursor.execute("ALTER TABLE model_stats ADD COLUMN feedback_comment TEXT")
            if "feedback_source" not in existing_columns:
                cursor.execute("ALTER TABLE model_stats ADD COLUMN feedback_source TEXT DEFAULT 'user'")
            if "feedback_user" not in existing_columns:
                cursor.execute("ALTER TABLE model_stats ADD COLUMN feedback_user TEXT")
            if "feedback_timestamp" not in existing_columns:
                cursor.execute("ALTER TABLE model_stats ADD COLUMN feedback_timestamp DATETIME")
            conn.commit()

    def _prune_old_records(self) -> None:
        retention_days = int(os.getenv("ROUTER_STATS_RETENTION_DAYS", "30"))
        if retention_days <= 0:
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM model_stats WHERE timestamp < datetime('now', ?)",
                (f"-{retention_days} days",),
            )
            cursor.execute(
                "DELETE FROM session_history WHERE timestamp < datetime('now', ?)",
                (f"-{retention_days} days",),
            )
            conn.commit()

    def log_request(self, stats: Dict) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_stats (
                    request_id, model_id, timestamp, input_chars, output_chars, 
                    duration_ms, cost, topic, session_id, format_score, sentiment_score, 
                    judge_score, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats["request_id"], stats["model_id"], datetime.now(),
                stats["input_chars"], stats["output_chars"], stats["duration_ms"],
                stats["cost"], stats["topic"], stats["session_id"],
                stats.get("format_score", 1.0), stats.get("sentiment_score", 0.0),
                stats.get("judge_score", 1.0), json.dumps(stats.get("embedding", []))
            ))
            conn.commit()

    def update_effectiveness(self, request_id: str, score: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE model_stats SET effectiveness_score = ? WHERE request_id = ?", (score, request_id))
            conn.commit()

    def update_quality_scores(
        self,
        request_id: str,
        format_score: float,
        density_score: float,
        judge_score: float,
        sentiment_score: float,
    ) -> None:
        combined_effectiveness = (judge_score * 0.5) + (format_score * 0.3) + (density_score * 0.2)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE model_stats
                SET format_score = ?, sentiment_score = ?, judge_score = ?, effectiveness_score = ?
                WHERE request_id = ?
                """,
                (format_score, sentiment_score, judge_score, combined_effectiveness, request_id),
            )
            conn.commit()

    def request_exists(self, request_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM model_stats WHERE request_id = ? LIMIT 1", (request_id,))
            return cursor.fetchone() is not None

    def apply_user_feedback(
        self,
        request_id: str,
        feedback_score: float,
        feedback_label: str,
        feedback_comment: str,
        feedback_source: str,
        feedback_user: str,
    ) -> Optional[dict[str, float]]:
        alpha = float(os.getenv("ROUTER_FEEDBACK_ALPHA", "0.35"))
        alpha = max(0.05, min(alpha, 0.95))
        score_clamped = max(-1.0, min(1.0, float(feedback_score)))
        target_effectiveness = (score_clamped + 1.0) / 2.0

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT effectiveness_score FROM model_stats WHERE request_id = ?",
                (request_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            current_effectiveness = float(row["effectiveness_score"] or 0.5)
            new_effectiveness = ((1.0 - alpha) * current_effectiveness) + (alpha * target_effectiveness)

            cursor.execute(
                """
                UPDATE model_stats
                SET effectiveness_score = ?,
                    feedback_score = ?,
                    feedback_label = ?,
                    feedback_comment = ?,
                    feedback_source = ?,
                    feedback_user = ?,
                    feedback_timestamp = ?
                WHERE request_id = ?
                """,
                (
                    new_effectiveness,
                    score_clamped,
                    feedback_label,
                    feedback_comment,
                    feedback_source,
                    feedback_user,
                    datetime.now(),
                    request_id,
                ),
            )
            conn.commit()
            return {
                "old_effectiveness": current_effectiveness,
                "new_effectiveness": new_effectiveness,
                "feedback_score": score_clamped,
            }

    def get_model_performance(self, model_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    AVG(effectiveness_score) as avg_effectiveness,
                    AVG(duration_ms) as avg_duration,
                    AVG(cost) as avg_cost,
                    AVG(format_score) as avg_format,
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(judge_score) as avg_judge
                FROM model_stats WHERE model_id = ?
            """, (model_id,))
            row = cursor.fetchone()
            if row and row["avg_effectiveness"] is not None:
                return dict(row)
            return None

    def log_session_input(self, session_id: str, text: str, embedding: list[float]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_history (session_id, timestamp, input_text, embedding_json)
                VALUES (?, ?, ?, ?)
            """, (session_id, datetime.now(), text, json.dumps(embedding)))
            conn.commit()

    def get_last_session_input(self, session_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM session_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (session_id,))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res["embedding"] = json.loads(res["embedding_json"])
                return res
            return None

    def penalize_last_request(self, session_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Reduce effectiveness of the latest request in the session
            cursor.execute("""
                UPDATE model_stats 
                SET effectiveness_score = effectiveness_score * 0.5 
                WHERE session_id = ? 
                AND timestamp = (SELECT MAX(timestamp) FROM model_stats WHERE session_id = ?)
            """, (session_id, session_id))
            conn.commit()

    def get_similar_performance(self, query_embedding: list[float], top_k: int = 50) -> dict[str, Dict]:
        """
        Calculates performance metrics per model for historical requests similar to the query.
        """
        import numpy as np
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch last 500 requests with embeddings
            cursor.execute("""
                SELECT model_id, duration_ms, cost, effectiveness_score, 
                       format_score, sentiment_score, judge_score, embedding_json
                FROM model_stats 
                ORDER BY timestamp DESC LIMIT 500
            """)
            rows = cursor.fetchall()
            
            if not rows:
                return {}
            
            # Calculate similarities
            similarities = []
            q_emb = np.array(query_embedding)
            
            for row in rows:
                try:
                    h_emb_json = row["embedding_json"]
                    if not h_emb_json:
                        continue
                    h_emb = np.array(json.loads(h_emb_json))
                    if h_emb.shape == q_emb.shape:
                        norm_q = np.linalg.norm(q_emb)
                        norm_h = np.linalg.norm(h_emb)
                        if norm_q > 0 and norm_h > 0:
                            sim = np.dot(q_emb, h_emb) / (norm_q * norm_h)
                            similarities.append((sim, row))
                except:
                    continue
            
            # Sort by similarity
            sorted_sims = sorted(similarities, key=lambda x: x[0], reverse=True)
            
            # Take top_k
            top_bound = min(len(sorted_sims), top_k)
            top_rows = []
            for i in range(top_bound):
                top_rows.append(sorted_sims[i][1])
            
            # Aggregate by model
            model_metrics = {}
            for row in top_rows:
                mid = row["model_id"]
                if mid not in model_metrics:
                    model_metrics[mid] = {
                        "count": 0, "eff": 0.0, "dur": 0.0, "cost": 0.0,
                        "fmt": 0.0, "sent": 0.0, "judge": 0.0
                    }
                s = model_metrics[mid]
                s["count"] += 1
                s["eff"] += row["effectiveness_score"]
                s["dur"] += row["duration_ms"]
                s["cost"] += row["cost"]
                s["fmt"] += row["format_score"]
                s["sent"] += row["sentiment_score"]
                s["judge"] += row["judge_score"]
            
            # Calculate averages
            results = {}
            for mid, s in model_metrics.items():
                results[mid] = {
                    "avg_effectiveness": s["eff"] / s["count"],
                    "avg_duration": s["dur"] / s["count"],
                    "avg_cost": s["cost"] / s["count"],
                    "avg_format": s["fmt"] / s["count"],
                    "avg_sentiment": s["sent"] / s["count"],
                    "avg_judge": s["judge"] / s["count"]
                }
            return results
