"""
Lost Event

Represents the moment an object disappears from a camera.
This object becomes the bridge between Tracking → ReID.
"""

from datetime import datetime, timezone


class LostEvent:
    def __init__(self, track, camera_id, timestamp=None):
        """
        Args:
            track: The LAST known Track object before disappearance
            camera_id: Camera where object was lost
            timestamp: When loss happened
        """

        self.track = track                      # <- REQUIRED for ReID
        self.track_id = track.id
        self.bbox = track.bbox
        self.label = getattr(track, "label", "unknown")

        self.camera_id = camera_id
        self.timestamp = timestamp or datetime.now(timezone.utc).timestamp()

    def __repr__(self):
        return (
            f"<LostEvent track_id={self.track_id} "
            f"camera={self.camera_id} "
            f"time={self.timestamp}>"
        )
