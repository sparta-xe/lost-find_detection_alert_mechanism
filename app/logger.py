"""
Professional logging system for the Lost Item Detection System.
"""
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)

class SystemLogger:
    """Professional logging system with multiple handlers and formatters."""
    
    def __init__(self, 
                 log_dir: str = "data/logs",
                 log_level: int = logging.INFO,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_database: bool = False):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory for log files
            log_level: Minimum logging level
            max_file_size: Maximum size per log file in bytes
            backup_count: Number of backup files to keep
            enable_console: Enable console logging
            enable_database: Enable database logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = log_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_database = enable_database
        
        # Initialize loggers
        self.setup_loggers()
    
    def setup_loggers(self):
        """Set up all logging handlers and formatters."""
        
        # Main system logger
        self.logger = logging.getLogger("LostItemSystem")
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File formatter
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console formatter
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        
        # Rotating file handler for general logs
        general_log_file = self.log_dir / "system.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error-only file handler
        error_log_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Performance logger for timing critical operations
        self.perf_logger = logging.getLogger("Performance")
        self.perf_logger.setLevel(logging.INFO)
        
        perf_log_file = self.log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        perf_formatter = logging.Formatter(
            fmt="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        perf_handler.setFormatter(perf_formatter)
        self.perf_logger.addHandler(perf_handler)
        
        # Detection logger for detection-specific events
        self.detection_logger = logging.getLogger("Detection")
        self.detection_logger.setLevel(logging.INFO)
        
        detection_log_file = self.log_dir / "detections.log"
        detection_handler = logging.handlers.RotatingFileHandler(
            detection_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        detection_handler.setFormatter(file_formatter)
        self.detection_logger.addHandler(detection_handler)
        
        # Matching logger for lost item matching events
        self.matching_logger = logging.getLogger("Matching")
        self.matching_logger.setLevel(logging.INFO)
        
        matching_log_file = self.log_dir / "matching.log"
        matching_handler = logging.handlers.RotatingFileHandler(
            matching_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        matching_handler.setFormatter(file_formatter)
        self.matching_logger.addHandler(matching_handler)
        
        self.logger.info("Logging system initialized")
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (if None, returns main system logger)
            
        Returns:
            Logger instance
        """
        if name is None:
            return self.logger
        elif name == "performance":
            return self.perf_logger
        elif name == "detection":
            return self.detection_logger
        elif name == "matching":
            return self.matching_logger
        else:
            # Create child logger
            child_logger = self.logger.getChild(name)
            return child_logger
    
    def log_performance(self, operation: str, duration: float, 
                       additional_info: Optional[dict] = None):
        """
        Log performance metrics.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            additional_info: Additional performance data
        """
        message = f"{operation}: {duration:.4f}s"
        if additional_info:
            info_str = ", ".join([f"{k}={v}" for k, v in additional_info.items()])
            message += f" ({info_str})"
        
        self.perf_logger.info(message)
    
    def log_detection(self, camera_id: str, detections_count: int, 
                     frame_number: int, processing_time: float):
        """
        Log detection events.
        
        Args:
            camera_id: Camera identifier
            detections_count: Number of detections found
            frame_number: Frame number
            processing_time: Processing time in seconds
        """
        self.detection_logger.info(
            f"Camera {camera_id}, Frame {frame_number}: "
            f"{detections_count} detections in {processing_time:.3f}s"
        )
    
    def log_match(self, lost_item_id: str, confidence: float, 
                 camera_id: str, match_reasons: list):
        """
        Log lost item match events.
        
        Args:
            lost_item_id: Lost item identifier
            confidence: Match confidence
            camera_id: Camera identifier
            match_reasons: List of match reasons
        """
        reasons_str = ", ".join(match_reasons) if match_reasons else "none"
        self.matching_logger.info(
            f"Match found: {lost_item_id} (confidence: {confidence:.3f}, "
            f"camera: {camera_id}, reasons: {reasons_str})"
        )
    
    def log_system_startup(self, config: dict):
        """
        Log system startup information.
        
        Args:
            config: System configuration dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("LOST ITEM DETECTION SYSTEM STARTUP")
        self.logger.info("=" * 60)
        
        for key, value in config.items():
            self.logger.info(f"Config: {key} = {value}")
        
        self.logger.info("System startup completed")
    
    def log_system_shutdown(self):
        """Log system shutdown."""
        self.logger.info("System shutdown initiated")
        self.logger.info("=" * 60)
    
    def create_session_logger(self, session_id: str) -> logging.Logger:
        """
        Create a session-specific logger.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session logger instance
        """
        session_logger = logging.getLogger(f"Session.{session_id}")
        session_logger.setLevel(self.log_level)
        
        # Session-specific log file
        session_log_file = self.log_dir / f"session_{session_id}.log"
        session_handler = logging.FileHandler(session_log_file, encoding='utf-8')
        
        session_formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        session_handler.setFormatter(session_formatter)
        session_logger.addHandler(session_handler)
        
        session_logger.info(f"Session {session_id} started")
        return session_logger

# Global logger instance
_system_logger = None

def setup_logger(log_dir: str = "data/logs", 
                log_level: int = logging.INFO,
                enable_console: bool = True) -> SystemLogger:
    """
    Set up the global logging system.
    
    Args:
        log_dir: Directory for log files
        log_level: Minimum logging level
        enable_console: Enable console logging
        
    Returns:
        SystemLogger instance
    """
    global _system_logger
    
    if _system_logger is None:
        _system_logger = SystemLogger(
            log_dir=log_dir,
            log_level=log_level,
            enable_console=enable_console
        )
    
    return _system_logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance from the global logging system.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _system_logger
    
    if _system_logger is None:
        _system_logger = setup_logger()
    
    return _system_logger.get_logger(name)

# Convenience functions for backward compatibility
def setup_basic_logger():
    """Set up basic logging configuration."""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "system.log"),
            logging.StreamHandler()
        ]
    )
