import uuid
from dataclasses import dataclass


@dataclass
class Track:
    id: str
    label: str
    bbox: tuple
    feature: object
    first_seen: float
    last_seen: float
    missed_frames: int = 0


class TrackMemory:
    """
    Stores all currently active tracks.
    """

    def __init__(self):
        self.active_tracks = {}

    def create_track(self, det, feature):
        track_id = str(uuid.uuid4())[:8]

        track = Track(
            id=track_id,
            label=det["label"],
            bbox=det["bbox"],
            feature=feature,
            first_seen=det["timestamp"],
            last_seen=det["timestamp"],
        )

        self.active_tracks[track_id] = track
        print(f"[NEW TRACK] {track_id} ({track.label})")

        return track

    def update_track(self, track_id, det, feature):
        track = self.active_tracks[track_id]
        track.bbox = det["bbox"]
        track.feature = feature
        track.last_seen = det["timestamp"]
        track.missed_frames = 0

    def mark_missed(self, track_id):
        self.active_tracks[track_id].missed_frames += 1

    def delete_track(self, track_id):
        print(f"[TRACK REMOVED] {track_id}")
        del self.active_tracks[track_id]
