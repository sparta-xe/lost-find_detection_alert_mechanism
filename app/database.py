"""
Professional database system for event logging and data persistence.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EventDatabase:
    """Professional database system for lost item detection events."""
    
    def __init__(self, db_path: str = "data/events.db"):
        """
        Initialize database connection and create tables.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize database with required tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Events table for general detection events
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                object_name TEXT,
                confidence REAL,
                bbox TEXT,
                timestamp TEXT NOT NULL,
                frame_number INTEGER,
                additional_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Lost items table
        c.execute("""
            CREATE TABLE IF NOT EXISTS lost_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                image_path TEXT,
                upload_timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                found_timestamp TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Matches table for lost item matches
        c.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT UNIQUE NOT NULL,
                lost_item_id TEXT NOT NULL,
                detection_id TEXT,
                camera_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox TEXT,
                frame_number INTEGER,
                timestamp TEXT NOT NULL,
                match_reasons TEXT,
                feature_scores TEXT,
                verified BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (lost_item_id) REFERENCES lost_items (item_id)
            )
        """)
        
        # Interactions table for person-object interactions
        c.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT UNIQUE NOT NULL,
                person_id TEXT,
                object_id TEXT,
                interaction_type TEXT NOT NULL,
                confidence REAL,
                duration REAL,
                camera_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                bbox_person TEXT,
                bbox_object TEXT,
                alert_level TEXT DEFAULT 'info',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System logs table
        c.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT,
                function TEXT,
                timestamp TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def log_event(self, camera_id: str, event_type: str, object_name: str = None,
                  confidence: float = None, bbox: Tuple = None, 
                  frame_number: int = None, additional_data: Dict = None) -> int:
        """
        Log a detection event to the database.
        
        Args:
            camera_id: Camera identifier
            event_type: Type of event (detection, lost_item_match, pickup_attempt, etc.)
            object_name: Name/label of detected object
            confidence: Detection confidence score
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            frame_number: Frame number in video
            additional_data: Additional event data as dictionary
            
        Returns:
            Event ID of inserted record
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        bbox_str = json.dumps(bbox) if bbox else None
        additional_str = json.dumps(additional_data) if additional_data else None
        
        c.execute("""
            INSERT INTO events 
            (camera_id, event_type, object_name, confidence, bbox, timestamp, 
             frame_number, additional_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (camera_id, event_type, object_name, confidence, bbox_str, 
              timestamp, frame_number, additional_str))
        
        event_id = c.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Logged event {event_id}: {event_type} on {camera_id}")
        return event_id
    
    def log_lost_item(self, item_id: str, name: str, description: str = None,
                     image_path: str = None) -> bool:
        """
        Log a new lost item registration.
        
        Args:
            item_id: Unique item identifier
            name: Item name
            description: Item description
            image_path: Path to item image
            
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            c.execute("""
                INSERT INTO lost_items (item_id, name, description, image_path, upload_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (item_id, name, description, image_path, timestamp))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Registered lost item: {item_id} ({name})")
            return True
            
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to register lost item {item_id}: {e}")
            return False
    
    def log_match(self, match_id: str, lost_item_id: str, detection_id: str,
                  camera_id: str, confidence: float, bbox: Tuple = None,
                  frame_number: int = None, match_reasons: List = None,
                  feature_scores: Dict = None) -> int:
        """
        Log a lost item match.
        
        Args:
            match_id: Unique match identifier
            lost_item_id: ID of matched lost item
            detection_id: Detection identifier
            camera_id: Camera identifier
            confidence: Match confidence score
            bbox: Bounding box coordinates
            frame_number: Frame number
            match_reasons: List of match reasons
            feature_scores: Dictionary of feature scores
            
        Returns:
            Match record ID
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        bbox_str = json.dumps(bbox) if bbox else None
        reasons_str = json.dumps(match_reasons) if match_reasons else None
        scores_str = json.dumps(feature_scores) if feature_scores else None
        
        c.execute("""
            INSERT INTO matches 
            (match_id, lost_item_id, detection_id, camera_id, confidence, bbox,
             frame_number, timestamp, match_reasons, feature_scores)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (match_id, lost_item_id, detection_id, camera_id, confidence,
              bbox_str, frame_number, timestamp, reasons_str, scores_str))
        
        match_record_id = c.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Logged match {match_id}: {lost_item_id} with confidence {confidence:.2f}")
        return match_record_id
    
    def log_interaction(self, interaction_id: str, person_id: str, object_id: str,
                       interaction_type: str, confidence: float, camera_id: str,
                       duration: float = None, bbox_person: Tuple = None,
                       bbox_object: Tuple = None, alert_level: str = "info") -> int:
        """
        Log a person-object interaction.
        
        Args:
            interaction_id: Unique interaction identifier
            person_id: Person identifier
            object_id: Object identifier
            interaction_type: Type of interaction
            confidence: Interaction confidence
            camera_id: Camera identifier
            duration: Interaction duration in seconds
            bbox_person: Person bounding box
            bbox_object: Object bounding box
            alert_level: Alert level (info, warning, critical)
            
        Returns:
            Interaction record ID
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        bbox_person_str = json.dumps(bbox_person) if bbox_person else None
        bbox_object_str = json.dumps(bbox_object) if bbox_object else None
        
        c.execute("""
            INSERT INTO interactions 
            (interaction_id, person_id, object_id, interaction_type, confidence,
             duration, camera_id, timestamp, bbox_person, bbox_object, alert_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (interaction_id, person_id, object_id, interaction_type, confidence,
              duration, camera_id, timestamp, bbox_person_str, bbox_object_str, alert_level))
        
        interaction_record_id = c.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Logged interaction {interaction_id}: {interaction_type}")
        return interaction_record_id
    
    def get_events(self, camera_id: str = None, event_type: str = None,
                   limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Retrieve events from database.
        
        Args:
            camera_id: Filter by camera ID
            event_type: Filter by event type
            limit: Maximum number of records
            offset: Record offset for pagination
            
        Returns:
            List of event dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        c.execute(query, params)
        events = [dict(row) for row in c.fetchall()]
        
        conn.close()
        return events
    
    def get_matches(self, lost_item_id: str = None, verified: bool = None,
                   limit: int = 100) -> List[Dict]:
        """
        Retrieve lost item matches.
        
        Args:
            lost_item_id: Filter by lost item ID
            verified: Filter by verification status
            limit: Maximum number of records
            
        Returns:
            List of match dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        query = "SELECT * FROM matches WHERE 1=1"
        params = []
        
        if lost_item_id:
            query += " AND lost_item_id = ?"
            params.append(lost_item_id)
        
        if verified is not None:
            query += " AND verified = ?"
            params.append(verified)
        
        query += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
        params.append(limit)
        
        c.execute(query, params)
        matches = [dict(row) for row in c.fetchall()]
        
        conn.close()
        return matches
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with various statistics
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        stats = {}
        
        # Total events
        c.execute("SELECT COUNT(*) FROM events")
        stats['total_events'] = c.fetchone()[0]
        
        # Total lost items
        c.execute("SELECT COUNT(*) FROM lost_items WHERE status = 'active'")
        stats['active_lost_items'] = c.fetchone()[0]
        
        # Total matches
        c.execute("SELECT COUNT(*) FROM matches")
        stats['total_matches'] = c.fetchone()[0]
        
        # Verified matches
        c.execute("SELECT COUNT(*) FROM matches WHERE verified = TRUE")
        stats['verified_matches'] = c.fetchone()[0]
        
        # Total interactions
        c.execute("SELECT COUNT(*) FROM interactions")
        stats['total_interactions'] = c.fetchone()[0]
        
        # Critical interactions
        c.execute("SELECT COUNT(*) FROM interactions WHERE alert_level = 'critical'")
        stats['critical_interactions'] = c.fetchone()[0]
        
        conn.close()
        return stats
    
    def mark_item_found(self, item_id: str) -> bool:
        """
        Mark a lost item as found.
        
        Args:
            item_id: Lost item identifier
            
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            c.execute("""
                UPDATE lost_items 
                SET status = 'found', found_timestamp = ?
                WHERE item_id = ?
            """, (timestamp, item_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Marked item {item_id} as found")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark item {item_id} as found: {e}")
            return False

# Convenience functions for backward compatibility
def init_db():
    """Initialize database with default path."""
    db = EventDatabase()
    return db

def log_event(camera_id: str, object_name: str, timestamp: str):
    """Simple event logging function."""
    db = EventDatabase()
    return db.log_event(camera_id, "detection", object_name)
