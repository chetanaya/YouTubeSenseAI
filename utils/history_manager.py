import os
import sqlite3
import datetime
import json
import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class VideoHistoryManager:
    """Manages video analysis history for different analysis methods"""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the history manager with optional custom DB path"""
        if db_path is None:
            # Create a .cache directory in the project root if it doesn't exist
            cache_dir = (
                Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                / ".cache"
            )
            os.makedirs(cache_dir, exist_ok=True)
            db_path = cache_dir / "video_history.db"

        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create the videos table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            video_url TEXT NOT NULL,
            title TEXT,
            timestamp DATETIME NOT NULL,
            analysis_method TEXT NOT NULL,
            metadata TEXT,
            UNIQUE(video_id, analysis_method)
        )
        """)

        conn.commit()
        conn.close()

    def add_video(
        self,
        video_id: str,
        video_url: str,
        analysis_method: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Add or update a video in the history

        Args:
            video_id: YouTube video ID
            video_url: Full YouTube URL
            analysis_method: Method used (youtube_api, whisper, assemblyai)
            title: Optional video title
            metadata: Optional additional data as dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Convert metadata to JSON string if provided
            metadata_json = json.dumps(metadata) if metadata else None

            # Use REPLACE to update if exists or insert if not
            cursor.execute(
                """
            INSERT OR REPLACE INTO video_history 
            (video_id, video_url, title, timestamp, analysis_method, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    video_id,
                    video_url,
                    title,
                    datetime.datetime.now().isoformat(),
                    analysis_method,
                    metadata_json,
                ),
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error adding video to history: {e}")
            return False

    def get_video_history(
        self, analysis_method: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """
        Get video history, optionally filtered by analysis method

        Args:
            analysis_method: Filter by this method if provided
            limit: Maximum number of results to return

        Returns:
            List of video history entries as dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()

            if analysis_method:
                cursor.execute(
                    """
                SELECT * FROM video_history 
                WHERE analysis_method = ? 
                ORDER BY timestamp DESC LIMIT ?
                """,
                    (analysis_method, limit),
                )
            else:
                cursor.execute(
                    """
                SELECT * FROM video_history 
                ORDER BY timestamp DESC LIMIT ?
                """,
                    (limit,),
                )

            rows = cursor.fetchall()
            conn.close()

            # Convert rows to dictionaries and parse metadata JSON
            results = []
            for row in rows:
                entry = dict(row)
                if entry.get("metadata"):
                    try:
                        entry["metadata"] = json.loads(entry["metadata"])
                    except:
                        entry["metadata"] = {}
                results.append(entry)

            return results
        except Exception as e:
            st.error(f"Error retrieving video history: {e}")
            return []

    def get_video_by_id(self, video_id: str, analysis_method: str) -> Optional[Dict]:
        """
        Get a specific video by ID and analysis method

        Args:
            video_id: YouTube video ID
            analysis_method: Analysis method to match

        Returns:
            Video entry as dictionary or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
            SELECT * FROM video_history 
            WHERE video_id = ? AND analysis_method = ?
            """,
                (video_id, analysis_method),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                entry = dict(row)
                if entry.get("metadata"):
                    try:
                        entry["metadata"] = json.loads(entry["metadata"])
                    except:
                        entry["metadata"] = {}
                return entry
            return None
        except Exception as e:
            st.error(f"Error retrieving video: {e}")
            return None

    def clear_history(self, analysis_method: Optional[str] = None) -> bool:
        """
        Clear video history, optionally for a specific analysis method only

        Args:
            analysis_method: Method to clear history for, or all if None

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if analysis_method:
                cursor.execute(
                    """
                DELETE FROM video_history WHERE analysis_method = ?
                """,
                    (analysis_method,),
                )
            else:
                cursor.execute("DELETE FROM video_history")

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error clearing history: {e}")
            return False


def render_video_history_widget(analysis_method: str):
    """
    Renders a video history widget for the sidebar

    Args:
        analysis_method: Which analysis method to show history for
    """
    method_names = {
        "youtube_api": "YouTube API",
        "whisper": "OpenAI Whisper",
        "assemblyai": "Assembly AI",
    }

    method_display_name = method_names.get(analysis_method, analysis_method)

    with st.sidebar:
        st.divider()
        st.markdown("### 📜 Video History")

        # Initialize history manager if not in session state
        if "history_manager" not in st.session_state:
            st.session_state.history_manager = VideoHistoryManager()

        # Get history for this method
        history = st.session_state.history_manager.get_video_history(
            analysis_method=analysis_method, limit=5
        )

        if not history:
            st.info(f"No previous videos analyzed with {method_display_name}")
            return

        st.markdown(f"**Recent videos analyzed with {method_display_name}:**")

        # Display videos as buttons
        for entry in history:
            video_title = entry.get("title") or f"Video {entry['video_id']}"
            # Truncate long titles
            if len(video_title) > 40:
                video_title = video_title[:37] + "..."

            # Format timestamp to be more friendly
            timestamp = datetime.datetime.fromisoformat(entry["timestamp"])
            friendly_time = timestamp.strftime("%b %d, %I:%M %p")

            if st.button(
                f"🎬 {video_title}\n📅 {friendly_time}", key=f"hist_{entry['id']}"
            ):
                # Set the URL in the query parameters to navigate to the video
                st.query_params["url"] = entry["video_url"]
                st.rerun()

        # Add a clear history button
        if st.button(f"🗑️ Clear History", key=f"clear_{analysis_method}"):
            if st.session_state.history_manager.clear_history(analysis_method):
                st.success(f"History cleared for {method_display_name}")
                st.rerun()
