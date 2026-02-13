"""
Professional configuration management system.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Detection configuration parameters."""
    confidence_threshold: float = 0.15
    iou_threshold: float = 0.3
    max_detections: int = 500
    image_size: int = 640
    enable_upscaling: bool = True
    enhance_contrast: bool = True
    multi_scale: bool = True

@dataclass
class MatchingConfig:
    """Lost item matching configuration."""
    threshold: float = 0.25
    template_weight: float = 0.5
    color_weight: float = 0.25
    keypoint_weight: float = 0.1
    shape_weight: float = 0.08
    texture_weight: float = 0.04
    edge_weight: float = 0.03

@dataclass
class TrackingConfig:
    """Object tracking configuration."""
    stationary_threshold: float = 3.0
    proximity_threshold: float = 100.0
    interaction_threshold: float = 50.0
    max_disappeared: int = 30
    max_distance: float = 100.0

@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    frame_skip: int = 1
    resize_factor: float = 1.0
    use_gpu: bool = True
    batch_size: int = 1
    num_workers: int = 4
    enable_tensorrt: bool = False

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "data/logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_database: bool = True

@dataclass
class SystemConfig:
    """Complete system configuration."""
    detection: DetectionConfig
    matching: MatchingConfig
    tracking: TrackingConfig
    performance: PerformanceConfig
    logging: LoggingConfig
    
    # System settings
    camera_id: str = "default_cam"
    video_source: Optional[str] = None
    output_dir: str = "output"
    enable_overlay: bool = True
    show_video: bool = False
    export_results: bool = False
    max_frames: Optional[int] = None

class ConfigManager:
    """Professional configuration management system."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration file paths
        self.default_config_path = self.config_dir / "default.yaml"
        self.user_config_path = self.config_dir / "user.yaml"
        self.runtime_config_path = self.config_dir / "runtime.json"
        
        # Create default configuration if it doesn't exist
        if not self.default_config_path.exists():
            self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration file."""
        default_config = SystemConfig(
            detection=DetectionConfig(),
            matching=MatchingConfig(),
            tracking=TrackingConfig(),
            performance=PerformanceConfig(),
            logging=LoggingConfig()
        )
        
        self.save_config(default_config, self.default_config_path)
        logger.info(f"Created default configuration: {self.default_config_path}")
    
    def load_config(self, config_path: Optional[str] = None) -> SystemConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (if None, loads default)
            
        Returns:
            SystemConfig instance
        """
        if config_path:
            config_file = Path(config_path)
        else:
            # Load user config if exists, otherwise default
            config_file = self.user_config_path if self.user_config_path.exists() else self.default_config_path
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}, using defaults")
            return SystemConfig(
                detection=DetectionConfig(),
                matching=MatchingConfig(),
                tracking=TrackingConfig(),
                performance=PerformanceConfig(),
                logging=LoggingConfig()
            )
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Convert to SystemConfig
            config = self._dict_to_config(config_data)
            logger.info(f"Loaded configuration from: {config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            logger.info("Using default configuration")
            return SystemConfig(
                detection=DetectionConfig(),
                matching=MatchingConfig(),
                tracking=TrackingConfig(),
                performance=PerformanceConfig(),
                logging=LoggingConfig()
            )
    
    def save_config(self, config: SystemConfig, config_path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            config: SystemConfig instance to save
            config_path: Path to save configuration (if None, saves to user config)
        """
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.user_config_path
        
        try:
            config_dict = asdict(config)
            
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Saved configuration to: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig."""
        # Extract sub-configurations
        detection_data = config_data.get('detection', {})
        matching_data = config_data.get('matching', {})
        tracking_data = config_data.get('tracking', {})
        performance_data = config_data.get('performance', {})
        logging_data = config_data.get('logging', {})
        
        # Create sub-config objects
        detection_config = DetectionConfig(**detection_data)
        matching_config = MatchingConfig(**matching_data)
        tracking_config = TrackingConfig(**tracking_data)
        performance_config = PerformanceConfig(**performance_data)
        logging_config = LoggingConfig(**logging_data)
        
        # Create main config
        system_config = SystemConfig(
            detection=detection_config,
            matching=matching_config,
            tracking=tracking_config,
            performance=performance_config,
            logging=logging_config
        )
        
        # Set system-level properties
        for key, value in config_data.items():
            if key not in ['detection', 'matching', 'tracking', 'performance', 'logging']:
                if hasattr(system_config, key):
                    setattr(system_config, key, value)
        
        return system_config
    
    def update_config(self, config: SystemConfig, updates: Dict[str, Any]) -> SystemConfig:
        """
        Update configuration with new values.
        
        Args:
            config: Current SystemConfig
            updates: Dictionary of updates
            
        Returns:
            Updated SystemConfig
        """
        config_dict = asdict(config)
        
        # Apply updates
        for key, value in updates.items():
            if '.' in key:
                # Handle nested keys like 'detection.confidence_threshold'
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        return self._dict_to_config(config_dict)
    
    def get_config_template(self) -> Dict[str, Any]:
        """
        Get configuration template with descriptions.
        
        Returns:
            Configuration template dictionary
        """
        return {
            "detection": {
                "confidence_threshold": {
                    "value": 0.15,
                    "description": "Minimum confidence for object detection",
                    "range": [0.05, 1.0]
                },
                "iou_threshold": {
                    "value": 0.3,
                    "description": "IoU threshold for non-maximum suppression",
                    "range": [0.1, 0.9]
                },
                "max_detections": {
                    "value": 500,
                    "description": "Maximum number of detections per frame",
                    "range": [50, 1000]
                },
                "image_size": {
                    "value": 640,
                    "description": "Input image size for YOLO model",
                    "options": [320, 416, 512, 640, 832, 1024]
                },
                "enable_upscaling": {
                    "value": True,
                    "description": "Enable upscaling for small object detection"
                },
                "enhance_contrast": {
                    "value": True,
                    "description": "Apply contrast enhancement preprocessing"
                },
                "multi_scale": {
                    "value": True,
                    "description": "Enable multi-scale detection"
                }
            },
            "matching": {
                "threshold": {
                    "value": 0.25,
                    "description": "Minimum confidence for lost item matches",
                    "range": [0.1, 1.0]
                },
                "template_weight": {
                    "value": 0.5,
                    "description": "Weight for template matching",
                    "range": [0.0, 1.0]
                },
                "color_weight": {
                    "value": 0.25,
                    "description": "Weight for color similarity",
                    "range": [0.0, 1.0]
                }
            },
            "tracking": {
                "stationary_threshold": {
                    "value": 3.0,
                    "description": "Time (seconds) before object is considered stationary",
                    "range": [1.0, 10.0]
                },
                "proximity_threshold": {
                    "value": 100.0,
                    "description": "Distance (pixels) for person-object proximity",
                    "range": [50.0, 200.0]
                }
            },
            "performance": {
                "frame_skip": {
                    "value": 1,
                    "description": "Process every Nth frame (1 = all frames)",
                    "range": [1, 10]
                },
                "resize_factor": {
                    "value": 1.0,
                    "description": "Resize factor for performance optimization",
                    "range": [0.5, 1.0]
                },
                "use_gpu": {
                    "value": True,
                    "description": "Use GPU acceleration if available"
                }
            }
        }
    
    def validate_config(self, config: SystemConfig) -> List[str]:
        """
        Validate configuration values.
        
        Args:
            config: SystemConfig to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate detection config
        if not 0.05 <= config.detection.confidence_threshold <= 1.0:
            errors.append("Detection confidence threshold must be between 0.05 and 1.0")
        
        if not 0.1 <= config.detection.iou_threshold <= 0.9:
            errors.append("IoU threshold must be between 0.1 and 0.9")
        
        # Validate matching config
        if not 0.1 <= config.matching.threshold <= 1.0:
            errors.append("Matching threshold must be between 0.1 and 1.0")
        
        # Validate weights sum to approximately 1.0
        total_weight = (config.matching.template_weight + config.matching.color_weight + 
                       config.matching.keypoint_weight + config.matching.shape_weight + 
                       config.matching.texture_weight + config.matching.edge_weight)
        
        if not 0.9 <= total_weight <= 1.1:
            errors.append(f"Matching weights should sum to ~1.0, got {total_weight:.3f}")
        
        # Validate tracking config
        if config.tracking.stationary_threshold < 0.5:
            errors.append("Stationary threshold must be at least 0.5 seconds")
        
        # Validate performance config
        if not 0.5 <= config.performance.resize_factor <= 1.0:
            errors.append("Resize factor must be between 0.5 and 1.0")
        
        if config.performance.frame_skip < 1:
            errors.append("Frame skip must be at least 1")
        
        return errors

# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration using global manager."""
    manager = get_config_manager()
    return manager.load_config(config_path)