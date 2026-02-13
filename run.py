#!/usr/bin/env python
"""
Professional CLI Entry Point for Lost Item Detection System

This is the main entry point for running the lost item detection system
from the command line with professional configuration management.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.logger import setup_logger, get_logger
from app.database import EventDatabase
from core.config import ConfigManager
from scripts.enhanced_tracking_v2 import EnhancedTrackingPipelineV2

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load system configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Default configuration
    return {
        "video_source": None,
        "camera_id": "default_cam",
        "detection_confidence": 0.15,
        "match_threshold": 0.25,
        "stationary_threshold": 3.0,
        "proximity_threshold": 100.0,
        "interaction_threshold": 50.0,
        "log_level": "INFO",
        "enable_gpu": True,
        "frame_skip": 1,
        "resize_factor": 1.0,
        "output_dir": "output",
        "enable_overlay": True,
        "enable_database": True,
        "show_video": False,
        "export_results": False
    }

def setup_system(config: Dict[str, Any]) -> tuple:
    """
    Initialize system components.
    
    Args:
        config: System configuration
        
    Returns:
        Tuple of (logger, database, pipeline)
    """
    # Setup logging
    log_level_map = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }
    
    log_level = log_level_map.get(config.get("log_level", "INFO"), 20)
    logger_system = setup_logger(
        log_dir="data/logs",
        log_level=log_level,
        enable_console=True
    )
    
    logger = get_logger("main")
    
    # Log system startup
    logger_system.log_system_startup(config)
    
    # Initialize database
    if config.get("enable_database", True):
        database = EventDatabase()
        logger.info("Database initialized")
    else:
        database = None
        logger.info("Database disabled")
    
    # Initialize pipeline
    pipeline = EnhancedTrackingPipelineV2(
        video_source=config["video_source"],
        camera_id=config.get("camera_id", "default_cam"),
        lost_item_threshold=config.get("match_threshold", 0.25),
        stationary_threshold=config.get("stationary_threshold", 3.0),
        proximity_threshold=config.get("proximity_threshold", 100.0),
        interaction_threshold=config.get("interaction_threshold", 50.0),
        simulate_realtime=config.get("simulate_realtime", False),
        log_level=log_level
    )
    
    logger.info("System initialization completed")
    return logger, database, pipeline

def run_pipeline(config: Dict[str, Any]):
    """
    Run the main detection pipeline.
    
    Args:
        config: System configuration
    """
    logger, database, pipeline = setup_system(config)
    
    try:
        # Add lost items if specified
        if config.get("lost_items"):
            for item_config in config["lost_items"]:
                success = pipeline.add_lost_item(
                    item_config["image_path"],
                    item_config["name"],
                    item_config.get("description", "")
                )
                if success:
                    logger.info(f"Added lost item: {item_config['name']}")
                    if database:
                        database.log_lost_item(
                            item_config.get("id", item_config["name"]),
                            item_config["name"],
                            item_config.get("description", ""),
                            item_config["image_path"]
                        )
                else:
                    logger.error(f"Failed to add lost item: {item_config['name']}")
        
        # Run processing
        logger.info("Starting video processing...")
        
        frame_count = 0
        total_detections = 0
        total_matches = 0
        
        for cam_id, timestamp, frame in pipeline.loader.frames():
            # Apply frame skipping for performance
            if frame_count % config.get("frame_skip", 1) != 0:
                frame_count += 1
                continue
            
            # Process frame
            objects, persons, interactions, lost_item_matches = pipeline.process_frame(frame, timestamp)
            
            # Update counters
            total_detections += len(objects) + len(persons)
            total_matches += len(lost_item_matches)
            
            # Log to database
            if database:
                # Log detections
                for obj in objects:
                    database.log_event(
                        cam_id, "object_detection", obj.label,
                        obj.confidence, obj.bbox, frame_count
                    )
                
                # Log matches
                for match in lost_item_matches:
                    database.log_match(
                        f"match_{frame_count}_{match.lost_item_id}",
                        match.lost_item_id,
                        match.detection_id,
                        cam_id,
                        match.confidence,
                        match.bbox,
                        frame_count,
                        match.match_reasons,
                        match.feature_scores
                    )
                
                # Log interactions
                for interaction in interactions:
                    alert_level = "critical" if interaction.interaction_type in ["pickup_attempt", "item_picked_up"] else "info"
                    database.log_interaction(
                        interaction.event_id,
                        interaction.person_id,
                        interaction.object_id,
                        interaction.interaction_type,
                        interaction.confidence,
                        cam_id,
                        interaction.duration,
                        interaction.bbox_person,
                        interaction.bbox_object,
                        alert_level
                    )
            
            # Show progress
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames, {total_detections} detections, {total_matches} matches")
            
            frame_count += 1
            
            # Check for early termination
            if config.get("max_frames") and frame_count >= config["max_frames"]:
                break
        
        # Final statistics
        logger.info(f"Processing completed: {frame_count} frames, {total_detections} detections, {total_matches} matches")
        
        # Export results if requested
        if config.get("export_results") and database:
            export_path = Path(config.get("output_dir", "output")) / f"results_{pipeline.camera_id}.json"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            results = {
                "summary": {
                    "frames_processed": frame_count,
                    "total_detections": total_detections,
                    "total_matches": total_matches,
                    "camera_id": cam_id
                },
                "events": database.get_events(cam_id),
                "matches": database.get_matches(),
                "statistics": database.get_statistics()
            }
            
            with open(export_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results exported to {export_path}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
    finally:
        # Cleanup
        pipeline.cleanup()
        logger_system = setup_logger()
        logger_system.log_system_shutdown()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Lost Item Detection System - Professional CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic video processing
  python run.py --video data/test_clips/sample.mp4
  
  # With lost item tracking
  python run.py --video data/test_clips/sample.mp4 --config config.json
  
  # Live camera processing
  python run.py --video 0 --camera-id "entrance_cam"
  
  # High performance mode
  python run.py --video sample.mp4 --frame-skip 2 --resize-factor 0.8
        """
    )
    
    # Video source
    parser.add_argument("--video", "-v", 
                       help="Video file path or camera index (0 for webcam)")
    
    # Configuration
    parser.add_argument("--config", "-c",
                       help="Configuration file path (JSON)")
    
    # Basic settings
    parser.add_argument("--camera-id", 
                       default="default_cam",
                       help="Camera identifier")
    
    parser.add_argument("--detection-confidence", 
                       type=float, default=0.15,
                       help="Detection confidence threshold")
    
    parser.add_argument("--match-threshold", 
                       type=float, default=0.25,
                       help="Lost item match threshold")
    
    # Performance settings
    parser.add_argument("--frame-skip", 
                       type=int, default=1,
                       help="Process every Nth frame")
    
    parser.add_argument("--resize-factor", 
                       type=float, default=1.0,
                       help="Resize factor for performance")
    
    parser.add_argument("--max-frames", 
                       type=int,
                       help="Maximum frames to process")
    
    # Output settings
    parser.add_argument("--output-dir", 
                       default="output",
                       help="Output directory")
    
    parser.add_argument("--export-results", 
                       action="store_true",
                       help="Export results to JSON")
    
    parser.add_argument("--show-video", 
                       action="store_true",
                       help="Show video with overlay")
    
    # Logging
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="Logging level")
    
    parser.add_argument("--no-database", 
                       action="store_true",
                       help="Disable database logging")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()
    
    # Override config with command line arguments
    if args.video:
        # Handle numeric camera index
        try:
            config["video_source"] = int(args.video)
        except ValueError:
            config["video_source"] = args.video
    
    config.update({
        "camera_id": args.camera_id,
        "detection_confidence": args.detection_confidence,
        "match_threshold": args.match_threshold,
        "frame_skip": args.frame_skip,
        "resize_factor": args.resize_factor,
        "max_frames": args.max_frames,
        "output_dir": args.output_dir,
        "export_results": args.export_results,
        "show_video": args.show_video,
        "log_level": args.log_level,
        "enable_database": not args.no_database
    })
    
    # Validate configuration
    if not config.get("video_source"):
        print("Error: No video source specified. Use --video to specify a video file or camera index.")
        sys.exit(1)
    
    # Run the system
    run_pipeline(config)

if __name__ == "__main__":
    main()