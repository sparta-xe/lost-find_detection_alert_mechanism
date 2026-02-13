TRACKABLE_OBJECTS = {
    "backpack",
    "handbag",
    "suitcase",
    "laptop",
    "cell phone",
}

MIN_CONFIDENCE = 0.25   # lowered
UNKNOWN_CONFIDENCE = 0.20


class ObjectFilter:

    def filter(self, detections):
        filtered = []

        for det in detections:

            # Accept strong known classes
            if det["label"] in TRACKABLE_OBJECTS and det["confidence"] > MIN_CONFIDENCE:
                filtered.append(det)
                continue

            # Accept weak detections that could be items (prototype heuristic)
            if det["confidence"] > UNKNOWN_CONFIDENCE and det["label"] not in ["person"]:
                det["label"] = "unknown_item"
                filtered.append(det)

        return filtered
