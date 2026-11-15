"""
storage_service.py — Pluggable persistent storage abstraction for EduLens MCP.
Separates raw conversation transcripts from derived machine learning analytics.
"""

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.mcp_server.config import settings


class StorageService(ABC):
    """Abstract interface defining required storage operations for EduLens MCP."""

    @abstractmethod
    def save_raw_session(self, session_data: Dict[str, Any]) -> str:
        """Saves raw conversation JSON and returns session_id."""
        pass

    @abstractmethod
    def get_raw_session(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves raw session by ID, scoped to user_id."""
        pass

    @abstractmethod
    def find_session_by_content_hash(self, content_hash: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Checks if identical conversation has already been ingested for user."""
        pass

    @abstractmethod
    def update_session_status(self, session_id: str, user_id: str, status: str, error_message: Optional[str] = None) -> None:
        """Updates ingestion and processing status of a raw session."""
        pass

    @abstractmethod
    def save_analytics(self, session_id: str, user_id: str, analytics_data: Dict[str, Any]) -> None:
        """Saves structured analytics results derived from ML pipeline."""
        pass

    @abstractmethod
    def get_analytics(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves analytics results for a session, scoped to user_id."""
        pass

    @abstractmethod
    def list_sessions(
        self,
        user_id: str,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None,
        class_filter: Optional[str] = None,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Returns total count and list of sessions matching filters."""
        pass

    @abstractmethod
    def get_student_summary(self, user_id: str) -> Dict[str, Any]:
        """Computes aggregate longitudinal metrics across all sessions for user."""
        pass

    @abstractmethod
    def get_topic_summary(self, user_id: str, topic: str) -> Optional[Dict[str, Any]]:
        """Computes aggregate performance metrics for a specific topic."""
        pass

    @abstractmethod
    def save_quiz_submission(self, quiz_data: Dict[str, Any]) -> str:
        """Saves evaluated quiz submission and returns quiz_id."""
        pass

    @abstractmethod
    def get_quiz_submission(self, quiz_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves quiz submission by ID."""
        pass

    @abstractmethod
    def list_quiz_submissions(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Returns total count and list of quiz submissions for user."""
        pass

    @abstractmethod
    def get_all_students(self) -> List[str]:
        """Returns list of distinct student/user IDs present in storage."""
        pass


class SQLiteStorageService(StorageService):
    """
    SQLite implementation of StorageService.
    Thread-safe, zero external infrastructure required, fully relational.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Creates tables and indexes if they do not exist."""
        with self._lock, self._connection() as conn:
            # 1. Raw Sessions Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                subject TEXT,
                topic TEXT,
                content_hash TEXT NOT NULL,
                turns_count INTEGER NOT NULL,
                raw_json TEXT NOT NULL,
                created_at TEXT,
                ingested_at TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_sessions_user ON raw_sessions(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_sessions_hash ON raw_sessions(user_id, content_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_sessions_topic ON raw_sessions(user_id, topic);")

            # 2. Analytics Results Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics_results (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                understanding_score REAL NOT NULL,
                confidence REAL NOT NULL,
                actual_quiz_score REAL,
                quiz_status TEXT NOT NULL,
                expressed_confidence_level TEXT NOT NULL,
                expressed_confidence_score REAL NOT NULL,
                overall_performance_score REAL,
                performance_category TEXT,
                discrepancy_detected INTEGER NOT NULL,
                discrepancy_details TEXT,
                pedagogical_feedback TEXT NOT NULL,
                full_analytics_json TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES raw_sessions(session_id) ON DELETE CASCADE
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_user ON analytics_results(user_id);")

            # 3. Quiz Submissions Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_submissions (
                quiz_id TEXT PRIMARY KEY,
                session_id TEXT,
                user_id TEXT NOT NULL,
                subject TEXT,
                topic TEXT,
                total_questions INTEGER NOT NULL,
                correct_count INTEGER NOT NULL,
                incorrect_count INTEGER NOT NULL,
                unanswered_count INTEGER NOT NULL,
                actual_score_percent REAL NOT NULL,
                question_breakdown_json TEXT NOT NULL,
                topic_breakdown_json TEXT NOT NULL,
                raw_submission_json TEXT NOT NULL,
                evaluated_at TEXT NOT NULL
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quiz_user ON quiz_submissions(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quiz_session ON quiz_submissions(session_id);")
            conn.commit()

    def save_raw_session(self, session_data: Dict[str, Any]) -> str:
        session_id = session_data["session_id"]
        with self._lock, self._connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO raw_sessions (
                session_id, user_id, source_agent, subject, topic,
                content_hash, turns_count, raw_json, created_at,
                ingested_at, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                session_id,
                session_data["user_id"],
                session_data.get("source_agent", "chatgpt"),
                session_data.get("subject"),
                session_data.get("topic"),
                session_data["content_hash"],
                session_data.get("turns_count", 0),
                json.dumps(session_data.get("raw_json", {}), ensure_ascii=False),
                session_data.get("created_at"),
                session_data.get("ingested_at", datetime.now(timezone.utc).isoformat()),
                session_data.get("status", "ingested"),
                session_data.get("error_message"),
            ))
            conn.commit()
        return session_id

    def get_raw_session(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM raw_sessions WHERE session_id = ? AND user_id = ?;",
                (session_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["raw_json"] = json.loads(data["raw_json"])
            return data

    def find_session_by_content_hash(self, content_hash: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM raw_sessions WHERE user_id = ? AND content_hash = ? ORDER BY ingested_at DESC LIMIT 1;",
                (user_id, content_hash),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["raw_json"] = json.loads(data["raw_json"])
            return data

    def update_session_status(self, session_id: str, user_id: str, status: str, error_message: Optional[str] = None) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE raw_sessions SET status = ?, error_message = ? WHERE session_id = ? AND user_id = ?;",
                (status, error_message, session_id, user_id),
            )
            conn.commit()

    def save_analytics(self, session_id: str, user_id: str, analytics_data: Dict[str, Any]) -> None:
        if "topics" in analytics_data and len(analytics_data["topics"]) > 0:
            top_topic = analytics_data["topics"][0]
            u_data = top_topic.get("understanding", {}) or {}
            q_data = top_topic.get("actual_quiz_performance") or {}
            c_data = top_topic.get("expressed_confidence", {}) or {}
            o_data = top_topic.get("overall_performance", {}) or {}
            d_list = o_data.get("discrepancies", [])
            pred_class = u_data.get("classification", "unknown")
            u_score = float(u_data.get("score_index", 0.0))
            conf = float(u_data.get("confidence", 0.0))
            ped_feedback = u_data.get("pedagogical_feedback", "")
            conf_level = c_data.get("classification", "unknown")
            conf_score = float(c_data.get("score", 0.0))
            act_quiz_score = q_data.get("actual_score_percent") if q_data else None
            quiz_status = q_data.get("status", "awaiting_quiz_submission") if q_data else "awaiting_quiz_submission"
            overall_score = o_data.get("score")
            perf_category = o_data.get("status")
            disc_detected = 1 if d_list else 0
            disc_details = "; ".join([str(d) for d in d_list]) if d_list else None
        else:
            u_data = analytics_data.get("topic_understanding", {}) or {}
            q_data = analytics_data.get("actual_quiz_performance", {}) or {}
            c_data = analytics_data.get("learning_confidence", {}) or {}
            o_data = analytics_data.get("overall_topic_performance", {}) or {}
            d_data = analytics_data.get("discrepancy_analysis", {}) or {}
            pred_class = u_data.get("predicted_class", "unknown")
            u_score = float(u_data.get("understanding_score", 0.0))
            conf = float(u_data.get("confidence", 0.0))
            ped_feedback = analytics_data.get("pedagogical_feedback", "")
            conf_level = c_data.get("expressed_confidence_level", "unknown")
            conf_score = float(c_data.get("expressed_confidence_score", 0.0))
            act_quiz_score = q_data.get("actual_score_percent")
            quiz_status = q_data.get("status", "awaiting_quiz_submission")
            overall_score = o_data.get("overall_performance_score")
            perf_category = o_data.get("performance_category")
            disc_detected = 1 if d_data.get("discrepancy_detected") else 0
            disc_details = d_data.get("details")

        with self._lock, self._connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO analytics_results (
                session_id, user_id, predicted_class, understanding_score,
                confidence, actual_quiz_score, quiz_status,
                expressed_confidence_level, expressed_confidence_score,
                overall_performance_score, performance_category,
                discrepancy_detected, discrepancy_details,
                pedagogical_feedback, full_analytics_json, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                session_id,
                user_id,
                pred_class,
                u_score,
                conf,
                act_quiz_score,
                quiz_status,
                conf_level,
                conf_score,
                overall_score,
                perf_category,
                disc_detected,
                disc_details,
                ped_feedback,
                json.dumps(analytics_data, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ))
            # Also update raw_session status to processed
            conn.execute(
                "UPDATE raw_sessions SET status = 'processed' WHERE session_id = ? AND user_id = ?;",
                (session_id, user_id),
            )
            conn.commit()

    def get_analytics(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM analytics_results WHERE session_id = ? AND user_id = ?;",
                (session_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["full_analytics_json"] = json.loads(data["full_analytics_json"])
            return data

    def list_sessions(
        self,
        user_id: str,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        search: Optional[str] = None,
        class_filter: Optional[str] = None,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        query = """
        SELECT r.session_id, r.user_id, r.subject, r.topic, r.source_agent,
               r.status, r.ingested_at, r.turns_count,
               a.predicted_class AS understanding_class,
               a.understanding_score,
               a.confidence AS model_confidence,
               a.actual_quiz_score,
               a.quiz_status,
               a.expressed_confidence_level,
               a.expressed_confidence_score,
               a.discrepancy_detected,
               a.pedagogical_feedback,
               a.overall_performance_score AS overall_score
        FROM raw_sessions r
        LEFT JOIN analytics_results a ON r.session_id = a.session_id
        WHERE r.user_id = ?
        """
        params: List[Any] = [user_id]

        if subject:
            query += " AND r.subject = ?"
            params.append(subject)
        if topic:
            query += " AND r.topic = ?"
            params.append(topic)
        if class_filter:
            query += " AND a.predicted_class = ?"
            params.append(class_filter)
        if search:
            query += " AND (r.topic LIKE ? OR r.subject LIKE ? OR r.session_id LIKE ?)"
            term = f"%{search.strip()}%"
            params.extend([term, term, term])
        if date_from:
            query += " AND r.ingested_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND r.ingested_at <= ?"
            params.append(date_to)

        # Count total
        count_query = f"SELECT COUNT(*) FROM ({query})"
        with self._lock, self._connection() as conn:
            total_count = conn.execute(count_query, params).fetchone()[0]

            query += " ORDER BY r.ingested_at DESC LIMIT ? OFFSET ?;"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            items = [dict(r) for r in rows]

        return total_count, items

    def get_student_summary(self, user_id: str) -> Dict[str, Any]:
        with self._lock, self._connection() as conn:
            # 1. Total sessions and distribution
            sessions_rows = conn.execute("""
            SELECT r.session_id, r.subject, r.topic, r.ingested_at,
                   a.predicted_class, a.understanding_score,
                   a.actual_quiz_score, a.expressed_confidence_score,
                   a.overall_performance_score
            FROM raw_sessions r
            LEFT JOIN analytics_results a ON r.session_id = a.session_id
            WHERE r.user_id = ?
            ORDER BY r.ingested_at ASC;
            """, (user_id,)).fetchall()

            total_sessions = len(sessions_rows)
            if total_sessions == 0:
                return {
                    "student_id": user_id,
                    "total_sessions": 0,
                    "topics_studied": [],
                    "understanding_distribution": {"understood": 0, "partial": 0, "confused": 0},
                    "average_understanding_score": 0.0,
                    "average_actual_quiz_score": None,
                    "average_confidence_score": 0.0,
                    "average_overall_score": None,
                    "topic_wise_breakdown": [],
                    "areas_of_confusion": [],
                    "learning_progress": [],
                }

            understanding_dist = {"understood": 0, "partial": 0, "confused": 0}
            u_scores = []
            q_scores = []
            c_scores = []
            o_scores = []
            topics_map: Dict[str, Dict[str, Any]] = {}
            progress: List[Dict[str, Any]] = []

            for row in sessions_rows:
                r = dict(row)
                cls = r.get("predicted_class")
                if cls in understanding_dist:
                    understanding_dist[cls] += 1

                if r.get("understanding_score") is not None:
                    u_scores.append(r["understanding_score"])
                if r.get("actual_quiz_score") is not None:
                    q_scores.append(r["actual_quiz_score"])
                if r.get("expressed_confidence_score") is not None:
                    c_scores.append(r["expressed_confidence_score"])
                if r.get("overall_performance_score") is not None:
                    o_scores.append(r["overall_performance_score"])

                t_name = r.get("topic") or "General"
                if t_name not in topics_map:
                    topics_map[t_name] = {
                        "topic": t_name,
                        "subject": r.get("subject"),
                        "sessions_count": 0,
                        "understood_count": 0,
                        "confused_count": 0,
                        "partial_count": 0,
                        "latest_understanding": cls,
                    }
                topics_map[t_name]["sessions_count"] += 1
                if cls == "understood":
                    topics_map[t_name]["understood_count"] += 1
                elif cls == "confused":
                    topics_map[t_name]["confused_count"] += 1
                elif cls == "partial":
                    topics_map[t_name]["partial_count"] += 1
                topics_map[t_name]["latest_understanding"] = cls

                progress.append({
                    "session_id": r["session_id"],
                    "date": r["ingested_at"],
                    "topic": t_name,
                    "understanding_class": cls,
                    "understanding_score": r.get("understanding_score"),
                    "quiz_score": r.get("actual_quiz_score"),
                    "overall_score": r.get("overall_performance_score"),
                })

            areas_of_confusion = [
                t for t, data in topics_map.items()
                if data["confused_count"] > 0 or data["partial_count"] > data["understood_count"]
            ]

            topic_breakdown = []
            for t, d in topics_map.items():
                n = d["sessions_count"]
                mastery_pct = round((d["understood_count"] / n) * 100, 1) if n > 0 else 0.0
                topic_breakdown.append({
                    "topic": t,
                    "subject": d["subject"],
                    "sessions": n,
                    "mastery_rate": mastery_pct,
                    "latest_understanding": d["latest_understanding"],
                })

            return {
                "student_id": user_id,
                "total_sessions": total_sessions,
                "topics_studied": sorted(list(topics_map.keys())),
                "understanding_distribution": understanding_dist,
                "average_understanding_score": round(sum(u_scores) / len(u_scores), 1) if u_scores else 0.0,
                "average_actual_quiz_score": round(sum(q_scores) / len(q_scores), 1) if q_scores else None,
                "average_confidence_score": round(sum(c_scores) / len(c_scores), 1) if c_scores else 0.0,
                "average_overall_score": round(sum(o_scores) / len(o_scores), 1) if o_scores else None,
                "topic_wise_breakdown": topic_breakdown,
                "areas_of_confusion": sorted(areas_of_confusion),
                "learning_progress": progress,
            }

    def get_topic_summary(self, user_id: str, topic: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute("""
            SELECT r.session_id, r.subject, r.topic, r.ingested_at,
                   a.predicted_class, a.understanding_score,
                   a.actual_quiz_score, a.expressed_confidence_score,
                   a.overall_performance_score
            FROM raw_sessions r
            LEFT JOIN analytics_results a ON r.session_id = a.session_id
            WHERE r.user_id = ? AND LOWER(r.topic) = LOWER(?)
            ORDER BY r.ingested_at ASC;
            """, (user_id, topic.strip())).fetchall()

            if not rows:
                return None

            dist = {"understood": 0, "partial": 0, "confused": 0}
            u_scores, q_scores, c_scores, o_scores = [], [], [], []
            trend = []
            subject = rows[0]["subject"]

            for row in rows:
                r = dict(row)
                cls = r.get("predicted_class")
                if cls in dist:
                    dist[cls] += 1
                if r.get("understanding_score") is not None:
                    u_scores.append(r["understanding_score"])
                if r.get("actual_quiz_score") is not None:
                    q_scores.append(r["actual_quiz_score"])
                if r.get("expressed_confidence_score") is not None:
                    c_scores.append(r["expressed_confidence_score"])
                if r.get("overall_performance_score") is not None:
                    o_scores.append(r["overall_performance_score"])

                trend.append({
                    "session_id": r["session_id"],
                    "date": r["ingested_at"],
                    "understanding_score": r.get("understanding_score"),
                    "quiz_score": r.get("actual_quiz_score"),
                    "overall_score": r.get("overall_performance_score"),
                })

            avg_u = round(sum(u_scores) / len(u_scores), 1) if u_scores else 0.0
            avg_q = round(sum(q_scores) / len(q_scores), 1) if q_scores else None
            avg_c = round(sum(c_scores) / len(c_scores), 1) if c_scores else 0.0
            avg_o = round(sum(o_scores) / len(o_scores), 1) if o_scores else None

            return {
                "topic": topic,
                "subject": subject,
                "student_id": user_id,
                "total_sessions": len(rows),
                "understanding_distribution": dist,
                "average_understanding_score": avg_u,
                "quiz_performance": {
                    "average_score_percent": avg_q,
                    "quizzes_taken": len(q_scores),
                },
                "expressed_confidence": {
                    "average_confidence_score": avg_c,
                },
                "overall_topic_performance": {
                    "composite_score": avg_o,
                } if avg_o is not None else None,
                "historical_trend": trend,
            }

    def save_quiz_submission(self, quiz_data: Dict[str, Any]) -> str:
        quiz_id = quiz_data["quiz_id"]
        with self._lock, self._connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO quiz_submissions (
                quiz_id, session_id, user_id, subject, topic,
                total_questions, correct_count, incorrect_count, unanswered_count,
                actual_score_percent, question_breakdown_json, topic_breakdown_json,
                raw_submission_json, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                quiz_id,
                quiz_data.get("session_id"),
                quiz_data["user_id"],
                quiz_data.get("subject"),
                quiz_data.get("topic"),
                quiz_data["total_questions"],
                quiz_data["correct_count"],
                quiz_data["incorrect_count"],
                quiz_data["unanswered_count"],
                quiz_data["actual_score_percent"],
                json.dumps(quiz_data.get("question_breakdown", []), ensure_ascii=False),
                json.dumps(quiz_data.get("topic_breakdown", {}), ensure_ascii=False),
                json.dumps(quiz_data.get("raw_submission", {}), ensure_ascii=False),
                quiz_data.get("evaluated_at", datetime.now(timezone.utc).isoformat()),
            ))
            conn.commit()
        return quiz_id

    def get_quiz_submission(self, quiz_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM quiz_submissions WHERE quiz_id = ? AND user_id = ?;",
                (quiz_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["question_breakdown"] = json.loads(data["question_breakdown_json"])
            data["topic_breakdown"] = json.loads(data["topic_breakdown_json"])
            data["raw_submission"] = json.loads(data["raw_submission_json"])
            return data

    def list_quiz_submissions(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        with self._lock, self._connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM quiz_submissions WHERE user_id = ?;", (user_id,)
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT quiz_id, session_id, user_id, subject, topic,
                       total_questions, correct_count, incorrect_count, unanswered_count,
                       actual_score_percent, evaluated_at
                FROM quiz_submissions
                WHERE user_id = ?
                ORDER BY evaluated_at DESC
                LIMIT ? OFFSET ?;
                """,
                (user_id, limit, offset),
            ).fetchall()
            return count, [dict(r) for r in rows]

    def get_all_students(self) -> List[str]:
        with self._lock, self._connection() as conn:
            users = set()
            for r in conn.execute("SELECT DISTINCT user_id FROM raw_sessions;"):
                if r[0]:
                    users.add(r[0])
            for r in conn.execute("SELECT DISTINCT user_id FROM quiz_submissions;"):
                if r[0]:
                    users.add(r[0])
            return sorted(list(users)) if users else ["student_001"]


# Global storage service instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = SQLiteStorageService()
    return _storage_service
