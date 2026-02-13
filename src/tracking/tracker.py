from .track_memory import TrackMemory
from .appearance import extract_feature, similarity


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """
    Hybrid Tracker:
    Uses BOTH spatial overlap (IOU) and appearance similarity.

    This prevents ID fragmentation when detection jitters.
    """

    def __init__(self, iou_threshold=0.3, appearance_threshold=0.7, max_missed=12):
        self.memory = TrackMemory()
        self.iou_threshold = iou_threshold
        self.appearance_threshold = appearance_threshold
        self.max_missed = max_missed

        # latest frame is injected externally (from test script)
        self.current_frame = None

    def update(self, detections, timestamp):
        matched_tracks = set()

        for det in detections:
            feature = extract_feature(self.current_frame, det["bbox"])

            best_track_id = None
            best_score = 0

            for track_id, track in self.memory.active_tracks.items():

                score_iou = iou(track.bbox, det["bbox"])
                score_app = similarity(track.feature, feature)

                score = max(score_iou, score_app)

                if score > best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is not None and (
                best_score > self.iou_threshold or best_score > self.appearance_threshold
            ):
                self.memory.update_track(best_track_id, det, feature)
                matched_tracks.add(best_track_id)

            else:
                new_track = self.memory.create_track(det, feature)
                matched_tracks.add(new_track.id)

        # Handle tracks not seen this frame
        lost_tracks = []

        for track_id in list(self.memory.active_tracks.keys()):
            if track_id not in matched_tracks:
                self.memory.mark_missed(track_id)

                if self.memory.active_tracks[track_id].missed_frames > self.max_missed:
                    lost_tracks.append(self.memory.active_tracks[track_id])
                    self.memory.delete_track(track_id)

        return lost_tracks
