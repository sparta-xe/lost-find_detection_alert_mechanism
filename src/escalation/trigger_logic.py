import math


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance(b1, b2):
    c1 = _center(b1)
    c2 = _center(b2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


class AbandonmentTrigger:
    """
    Converts tracking signals into a semantic alert.

    Logic:
    Object + Person nearby  → OK
    Object alone for X sec → ABANDONED
    """

    def __init__(self, proximity_px=120, abandon_time=3.0):
        self.proximity_px = proximity_px
        self.abandon_time = abandon_time

        # track_id → last time a person was seen nearby
        self.last_attended = {}

    def update(self, active_tracks, person_detections, timestamp):
        alerts = []

        for track in active_tracks:
            nearest_person_dist = None

            # Find closest person
            for person in person_detections:
                d = _distance(track.bbox, person["bbox"])
                if nearest_person_dist is None or d < nearest_person_dist:
                    nearest_person_dist = d

            # If a person is close → object is attended
            if nearest_person_dist is not None and nearest_person_dist < self.proximity_px:
                self.last_attended[track.id] = timestamp
                continue

            # No person nearby → check how long it's been unattended
            last_seen = self.last_attended.get(track.id, track.first_seen)

            if (timestamp - last_seen) > self.abandon_time:
                alerts.append(track)

        return alerts
