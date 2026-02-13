#!/usr/bin/env python
"""
Interactive Streamlit Application for Lost Item Detection and Tracking

This application provides a web-based interface for:
- Uploading lost item images
- Real-time video processing with overlay
- Interactive lost item matching
- Live statistics and alerts
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.video_loader import VideoLoader
from src.detection.yolo_detector import YOLODetector
from src.detection.object_filter import ObjectFilter
from src.tracking.tracker import SimpleTracker
from src.escalation.lost_item_service import LostItemService, LostItemReporter
from scripts.enhanced_tracking import EnhancedTrackingPipeline
from scripts.enhanced_tracking_v2 import EnhancedTrackingPipelineV2


# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Lost Item Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .alert-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
    .alert-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'lost_item_service' not in st.session_state:
        st.session_state.lost_item_service = LostItemService()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'frames_processed': 0,
            'detections': 0,
            'matches': 0,
            'lost_items': 0,
            'pickup_attempts': 0,
            'items_picked_up': 0,
            'stationary_objects': 0,
            'active_alerts': 0
        }
    if 'matches_history' not in st.session_state:
        st.session_state.matches_history = []
    if 'active_alerts' not in st.session_state:
        st.session_state.active_alerts = []
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def draw_detection_overlay(frame: np.ndarray, objects: List, persons: List = None,
                          interactions: List = None, matches: List = None) -> np.ndarray:
    """Draw enhanced detection boxes and interactions on frame."""
    overlay_frame = frame.copy()
    
    # Draw objects
    for obj in objects:
        if hasattr(obj, 'bbox'):
            bbox = obj.bbox
            label = getattr(obj, 'label', 'object')
            state = getattr(obj, 'state', 'unknown')
            pickup_attempt = getattr(obj, 'pickup_attempt', False)
            person_nearby = getattr(obj, 'person_nearby', False)
        else:
            # Fallback for regular detections
            bbox = obj.get('bbox')
            label = obj.get('label', 'unknown')
            state = 'normal'
            pickup_attempt = False
            person_nearby = False
        
        if bbox is None:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color based on state
        if pickup_attempt:
            color = (0, 0, 255)    # Red for pickup attempt
        elif state == 'stationary':
            color = (0, 255, 255)  # Yellow for stationary
        elif state == 'dropped':
            color = (0, 165, 255)  # Orange for dropped
        else:
            color = (0, 255, 0)    # Green for normal
        
        # Draw bounding box
        thickness = 3 if pickup_attempt else 2
        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with state
        label_text = f"{label}"
        if hasattr(obj, 'state'):
            label_text += f" ({state})"
        if person_nearby:
            label_text += " [Person nearby]"
        if pickup_attempt:
            label_text += " [PICKUP ATTEMPT!]"
        
        cv2.putText(overlay_frame, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw persons
    if persons:
        for person in persons:
            x1, y1, x2, y2 = map(int, person.bbox)
            
            # Color based on behavior
            color = (255, 0, 0) if person.suspicious_behavior else (255, 255, 0)
            
            # Draw bounding box
            thickness = 3 if person.suspicious_behavior else 2
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"Person {person.person_id}"
            if person.suspicious_behavior:
                label += " [SUSPICIOUS]"
            
            cv2.putText(overlay_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw lost item matches with special highlighting
    if matches:
        for match in matches:
            bbox = match.get('bbox') if isinstance(match, dict) else match.bbox
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw highlighted bounding box for matches
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Draw match label with background
            item_name = match.get('lost_item_id') if isinstance(match, dict) else match.lost_item_id
            confidence = match.get('confidence', 0) if isinstance(match, dict) else match.confidence
            match_text = f"FOUND: {item_name} ({confidence:.1%})"
            
            text_size = cv2.getTextSize(match_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(overlay_frame, (x1, y1-30), (x1+text_size[0]+10, y1), (0, 0, 255), -1)
            cv2.putText(overlay_frame, match_text, (x1+5, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw interactions
    if interactions:
        for interaction in interactions:
            if hasattr(interaction, 'bbox_person') and hasattr(interaction, 'bbox_object'):
                if interaction.bbox_person and interaction.bbox_object:
                    # Draw line between person and object
                    person_center = ((interaction.bbox_person[0] + interaction.bbox_person[2]) // 2,
                                   (interaction.bbox_person[1] + interaction.bbox_person[3]) // 2)
                    object_center = ((interaction.bbox_object[0] + interaction.bbox_object[2]) // 2,
                                   (interaction.bbox_object[1] + interaction.bbox_object[3]) // 2)
                    
                    color = (0, 0, 255) if interaction.interaction_type == 'pickup_attempt' else (255, 0, 255)
                    cv2.line(overlay_frame, person_center, object_center, color, 3)
                    
                    # Draw interaction label
                    mid_point = ((person_center[0] + object_center[0]) // 2,
                               (person_center[1] + object_center[1]) // 2)
                    cv2.putText(overlay_frame, interaction.interaction_type.upper(), 
                               mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return overlay_frame


def format_timestamp(timestamp: float) -> str:
    """Format timestamp for display."""
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


# ============================================================================
# STREAMLIT COMPONENTS
# ============================================================================

def render_header():
    """Render application header."""
    st.markdown('<h1 class="main-header">🔍 Lost Item Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render sidebar with controls and lost item management."""
    st.sidebar.header("🎛️ Controls")
    
    # Video source selection
    st.sidebar.subheader("📹 Video Source")
    video_source = st.sidebar.selectbox(
        "Select video source:",
        ["Upload Video", "Use Sample Video", "Webcam (Live)"]
    )
    
    video_file = None
    if video_source == "Upload Video":
        video_file = st.sidebar.file_uploader(
            "Upload video file", 
            type=['mp4', 'avi', 'mov', 'mkv']
        )
    elif video_source == "Use Sample Video":
        sample_videos = list(Path("data/test_clips").glob("*.mp4")) if Path("data/test_clips").exists() else []
        if sample_videos:
            selected_sample = st.sidebar.selectbox(
                "Select sample video:",
                [v.name for v in sample_videos]
            )
            video_file = f"data/test_clips/{selected_sample}"
        else:
            st.sidebar.warning("No sample videos found in data/test_clips/")
            video_file = None
    else:
        # Webcam case
        video_file = 0
    
    # Processing parameters
    st.sidebar.subheader("⚙️ Detection Settings")
    detection_conf = st.sidebar.slider("Detection Confidence", 0.05, 1.0, 0.15, 0.05)
    match_threshold = st.sidebar.slider("Lost Item Match Threshold", 0.1, 1.0, 0.25, 0.05)
    
    # Enhanced detection parameters
    st.sidebar.subheader("🔍 Enhanced Detection")
    stationary_threshold = st.sidebar.slider("Stationary Time (seconds)", 1.0, 10.0, 3.0, 0.5)
    proximity_threshold = st.sidebar.slider("Person-Object Proximity (pixels)", 50, 200, 100, 10)
    interaction_threshold = st.sidebar.slider("Interaction Distance (pixels)", 20, 100, 50, 5)
    
    # Small object detection
    st.sidebar.subheader("🔬 Small Object Detection")
    enable_upscaling = st.sidebar.checkbox("Enable Upscaling for Small Objects", True)
    enhance_contrast = st.sidebar.checkbox("Enhance Contrast", True)
    
    max_frames = st.sidebar.number_input("Max Frames (0 = unlimited)", 0, 10000, 0)
    
    # Detection tips
    st.sidebar.info("""
    💡 **Tips for Better Detection:**
    - Lower detection confidence for small items
    - Lower match threshold for similar items (try 0.2-0.3 for screenshots)
    - Enable upscaling for very small objects
    - Use clear, well-lit reference images
    - **For screenshot matching:** Use threshold 0.2-0.3
    """)
    
    # Screenshot matching tip
    st.sidebar.success("""
    📸 **Screenshot Matching:**
    If you're uploading a screenshot from the same video, 
    set the match threshold to 0.25 or lower for better results.
    """)
    
    # Lost item management
    st.sidebar.subheader("📦 Lost Items")
    
    # Upload lost item
    uploaded_item = st.sidebar.file_uploader(
        "Upload lost item image", 
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_item:
        item_name = st.sidebar.text_input("Item Name", "")
        item_description = st.sidebar.text_area("Description (optional)", "")
        
        if st.sidebar.button("Add Lost Item") and item_name:
            # Save uploaded image
            item_path = f"data/lost_items/{uploaded_item.name}"
            Path("data/lost_items").mkdir(exist_ok=True)
            
            with open(item_path, "wb") as f:
                f.write(uploaded_item.getbuffer())
            
            # Add to service
            success, result = st.session_state.lost_item_service.upload_lost_item(
                item_path, item_name, item_description
            )
            
            if success:
                st.sidebar.success(f"✅ Added: {item_name}")
            else:
                st.sidebar.error(f"❌ Failed to add item")
    
    # Display registered items
    lost_items = st.session_state.lost_item_service.get_lost_items()
    if lost_items:
        st.sidebar.write(f"**Registered Items ({len(lost_items)}):**")
        for item in lost_items:
            st.sidebar.write(f"• {item['name']}")
    
    return {
        'video_source': video_source,
        'video_file': video_file,
        'detection_conf': detection_conf,
        'match_threshold': match_threshold,
        'stationary_threshold': stationary_threshold,
        'proximity_threshold': proximity_threshold,
        'interaction_threshold': interaction_threshold,
        'enable_upscaling': enable_upscaling,
        'enhance_contrast': enhance_contrast,
        'max_frames': max_frames if max_frames > 0 else None
    }


def render_main_content(config: Dict):
    """Render main content area with video processing."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📺 Video Processing")
        
        # Video display area
        video_placeholder = st.empty()
        
        # Control buttons
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            # Check if we have a valid video source
            has_video = (
                (config['video_source'] == "Upload Video" and config['video_file'] is not None) or
                (config['video_source'] == "Use Sample Video" and config['video_file'] is not None) or
                (config['video_source'] == "Webcam (Live)")
            )
            start_processing = st.button("▶️ Start Processing", disabled=st.session_state.processing or not has_video)
        
        with button_col2:
            stop_processing = st.button("⏹️ Stop Processing", disabled=not st.session_state.processing)
        
        with button_col3:
            clear_results = st.button("🗑️ Clear Results")
        
        # Processing logic
        if start_processing and has_video:
            st.session_state.processing = True
            process_video(config, video_placeholder)
        elif start_processing and not has_video:
            st.error("❌ Please select a video source first")
        
        if not has_video:
            st.info("📹 Please select a video source from the sidebar to begin processing")
        
        if stop_processing:
            st.session_state.processing = False
        
        if clear_results:
            st.session_state.stats = {'frames_processed': 0, 'detections': 0, 'matches': 0, 'lost_items': 0}
            st.session_state.matches_history = []
    
    with col2:
        render_statistics_panel()


def render_statistics_panel():
    """Render real-time statistics panel."""
    st.subheader("📊 Live Statistics")
    
    # Metrics in a 2x4 grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Frames Processed", st.session_state.stats['frames_processed'])
        st.metric("Total Detections", st.session_state.stats['detections'])
    
    with col2:
        st.metric("Lost Item Matches", st.session_state.stats['matches'])
        st.metric("Pickup Attempts", st.session_state.stats.get('pickup_attempts', 0))
    
    with col3:
        st.metric("Items Picked Up", st.session_state.stats.get('items_picked_up', 0))
        st.metric("Stationary Objects", st.session_state.stats.get('stationary_objects', 0))
    
    with col4:
        st.metric("Registered Items", st.session_state.stats['lost_items'])
        st.metric("Active Alerts", st.session_state.stats.get('active_alerts', 0))
    
    # Recent matches and alerts
    if st.session_state.matches_history:
        st.subheader("🎯 Recent Activity")
        
        # Convert to DataFrame for better display
        df_matches = pd.DataFrame(st.session_state.matches_history[-10:])  # Last 10 matches
        if not df_matches.empty:
            df_matches['timestamp'] = df_matches['timestamp'].apply(format_timestamp)
            st.dataframe(df_matches[['timestamp', 'item_name', 'confidence']], use_container_width=True)
    
    # Active alerts section
    st.subheader("🚨 Active Alerts")
    if 'active_alerts' not in st.session_state:
        st.session_state.active_alerts = []
    
    if st.session_state.active_alerts:
        for alert in st.session_state.active_alerts[-5:]:  # Show last 5 alerts
            alert_type = alert.get('type', 'info')
            message = alert.get('message', 'Unknown alert')
            
            if alert_type == 'item_picked_up':
                st.error(f"🚨 {message}")
            elif alert_type == 'pickup_attempt':
                st.warning(f"⚠️ {message}")
            else:
                st.info(f"ℹ️ {message}")
    else:
        st.info("No active alerts")
    
    # Lost items list
    lost_items = st.session_state.lost_item_service.get_lost_items()
    if lost_items:
        st.subheader("📦 Registered Items")
        for item in lost_items:
            with st.expander(f"📦 {item['name']}"):
                st.write(f"**Description:** {item.get('description', 'N/A')}")
                st.write(f"**Added:** {item.get('timestamp', 'N/A')}")
                
                # Show item image if available
                if 'image_path' in item and Path(item['image_path']).exists():
                    img = Image.open(item['image_path'])
                    st.image(img, width=150)


def process_video(config: Dict, video_placeholder):
    """Process video with real-time overlay and lost item detection."""
    try:
        # Initialize pipeline
        if config['video_source'] == "Webcam (Live)":
            video_source = 0  # Default webcam
        elif config['video_source'] == "Use Sample Video":
            video_source = config['video_file']
        else:
            # Handle uploaded file
            if hasattr(config['video_file'], 'name'):
                # It's a Streamlit uploaded file
                # Save it temporarily
                temp_path = f"temp_{config['video_file'].name}"
                with open(temp_path, "wb") as f:
                    f.write(config['video_file'].getbuffer())
                video_source = temp_path
            else:
                video_source = config['video_file']
        
        if not video_source:
            st.error("❌ No video source selected")
            return
        
        # Create pipeline
        try:
            pipeline = EnhancedTrackingPipelineV2(
                video_source=video_source,
                camera_id="streamlit_cam",
                lost_item_threshold=config['match_threshold'],
                stationary_threshold=config['stationary_threshold'],
                proximity_threshold=config['proximity_threshold'],
                interaction_threshold=config['interaction_threshold'],
                simulate_realtime=False  # Process as fast as possible for Streamlit
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize video source: {str(e)}")
            st.info("💡 Try a different video source or check camera permissions")
            return
        
        # Add existing lost items to pipeline
        lost_items = st.session_state.lost_item_service.get_lost_items()
        for item in lost_items:
            if 'image_path' in item and Path(item['image_path']).exists():
                pipeline.add_lost_item(
                    item['image_path'], 
                    item['name'], 
                    item.get('description', '')
                )
        
        st.session_state.stats['lost_items'] = len(lost_items)
        
        # Process frames
        frame_count = 0
        for cam_id, timestamp, frame in pipeline.loader.frames():
            if not st.session_state.processing:
                break
            
            # Process frame with enhanced detection
            objects, persons, interactions, lost_item_matches = pipeline.process_frame(frame, timestamp)
            
            # Update statistics
            st.session_state.stats['frames_processed'] += 1
            st.session_state.stats['detections'] += len(objects) + len(persons)
            st.session_state.stats['matches'] += len(lost_item_matches)
            
            # Add matches to history
            for match in lost_item_matches:
                match_data = {
                    'timestamp': timestamp,
                    'item_name': match.lost_item_id if hasattr(match, 'lost_item_id') else match.get('lost_item_id', 'Unknown'),
                    'confidence': f"{match.confidence:.1%}" if hasattr(match, 'confidence') else f"{match.get('confidence', 0):.1%}",
                    'camera': cam_id
                }
                st.session_state.matches_history.append(match_data)
            
            # Add interaction alerts
            for interaction in interactions:
                if interaction.interaction_type == 'pickup_attempt':
                    alert_data = {
                        'type': 'pickup_attempt',
                        'timestamp': timestamp,
                        'item_name': f"Pickup attempt on {interaction.object_id}",
                        'confidence': f"{interaction.confidence:.1%}",
                        'camera': cam_id,
                        'message': f"Person {interaction.person_id} attempting to take {interaction.object_id}"
                    }
                    st.session_state.matches_history.append(alert_data)
                    st.session_state.active_alerts.append(alert_data)
                
                elif interaction.interaction_type == 'item_picked_up':
                    alert_data = {
                        'type': 'item_picked_up',
                        'timestamp': timestamp,
                        'item_name': f"Item picked up: {interaction.object_id}",
                        'confidence': f"{interaction.confidence:.1%}",
                        'camera': cam_id,
                        'message': f"Person {interaction.person_id} has taken {interaction.object_id}"
                    }
                    st.session_state.matches_history.append(alert_data)
                    st.session_state.active_alerts.append(alert_data)
            
            # Create overlay with enhanced information
            overlay_frame = draw_detection_overlay(frame, objects, persons, interactions, lost_item_matches)
            
            # Convert to RGB for Streamlit
            overlay_frame_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(overlay_frame_rgb, channels="RGB", use_column_width=True)
            
            # Show alerts
            if lost_item_matches:
                st.success(f"🎯 **LOST ITEM FOUND!** {len(lost_item_matches)} match(es) detected")
            
            if interactions:
                pickup_attempts = [i for i in interactions if i.interaction_type == 'pickup_attempt']
                items_picked_up = [i for i in interactions if i.interaction_type == 'item_picked_up']
                
                if pickup_attempts:
                    st.error(f"🚨 **PICKUP ATTEMPT DETECTED!** {len(pickup_attempts)} attempt(s)")
                
                if items_picked_up:
                    st.error(f"� **IsTEM PICKED UP!** {len(items_picked_up)} item(s) taken")
                    for item in items_picked_up:
                        st.error(f"Person {item.person_id} took {item.object_id}")
            
            # Show object states
            stationary_objects = [o for o in objects if hasattr(o, 'state') and o.state == 'stationary']
            picked_up_objects = [o for o in objects if hasattr(o, 'state') and o.state == 'picked_up']
            
            if stationary_objects:
                st.warning(f"📦 {len(stationary_objects)} stationary/dropped object(s) detected")
            
            if picked_up_objects:
                st.error(f"🚨 {len(picked_up_objects)} object(s) have been picked up!")
                for obj in picked_up_objects:
                    if hasattr(obj, 'picked_up_by') and obj.picked_up_by:
                        st.error(f"• {obj.label} taken by Person {obj.picked_up_by}")
            
            # Show person tracking info
            suspicious_persons = [p for p in persons if p.suspicious_behavior]
            if suspicious_persons:
                st.warning(f"👤 {len(suspicious_persons)} suspicious person(s) detected")
                for person in suspicious_persons:
                    if hasattr(person, 'carrying_objects') and person.carrying_objects:
                        st.warning(f"• Person {person.person_id} carrying {len(person.carrying_objects)} object(s)")
            
            # Update session stats with new metrics
            st.session_state.stats['pickup_attempts'] = st.session_state.stats.get('pickup_attempts', 0) + len([i for i in interactions if i.interaction_type == 'pickup_attempt'])
            st.session_state.stats['items_picked_up'] = st.session_state.stats.get('items_picked_up', 0) + len([i for i in interactions if i.interaction_type == 'item_picked_up'])
            st.session_state.stats['stationary_objects'] = len(stationary_objects)
            
            # Limit processing rate for Streamlit
            time.sleep(0.1)
            
            # Check frame limit
            if config['max_frames'] and frame_count >= config['max_frames']:
                break
            
            frame_count += 1
        
        # Cleanup
        pipeline.cleanup()
        st.session_state.processing = False
        
        # Show final statistics
        st.info("✅ Processing completed!")
        
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        st.session_state.processing = False


def render_results_tab():
    """Render results and export tab."""
    st.subheader("📋 Results & Export")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Generate Report"):
            if st.session_state.lost_item_service.get_lost_items():
                reporter = LostItemReporter(st.session_state.lost_item_service)
                report = reporter.report_matches()
                st.text_area("Report", report, height=300)
            else:
                st.warning("No lost items registered")
    
    with col2:
        if st.button("💾 Export Results"):
            if st.session_state.matches_history:
                # Export matches history
                export_data = {
                    'statistics': st.session_state.stats,
                    'matches': st.session_state.matches_history,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"lost_item_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No results to export")
    
    # Matches history table
    if st.session_state.matches_history:
        st.subheader("🎯 All Matches")
        df = pd.DataFrame(st.session_state.matches_history)
        if not df.empty:
            df['timestamp'] = df['timestamp'].apply(format_timestamp)
            st.dataframe(df, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    render_header()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎥 Live Processing", "📊 Results & Export"])
    
    with tab1:
        config = render_sidebar()
        render_main_content(config)
    
    with tab2:
        render_results_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Lost Item Detection System | Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()