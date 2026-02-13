import cv2
import numpy as np

def extract_feature(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (64, 64))

    hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def similarity(f1, f2):
    if f1 is None or f2 is None:
        return 0
    return cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL)
