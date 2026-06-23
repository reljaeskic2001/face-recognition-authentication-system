import cv2
import numpy as np
from deepface import DeepFace

def deepface_attrs(bgr_crop):
    result = DeepFace.analyze(
        bgr_crop,
        actions=["age", "gender", "emotion"],
        enforce_detection=False,
        detector_backend="skip"
    )[0]

    return {
        "age": int(result["age"]),
        "gender": result["dominant_gender"],
        "expression": result["dominant_emotion"]
    }

def detect_glasses_simple(bgr_crop):
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # approximate eye/glasses region
    y1, y2 = int(h * 0.25), int(h * 0.50)
    x1, x2 = int(w * 0.15), int(w * 0.85)

    eye_region = gray[y1:y2, x1:x2]

    edges = cv2.Canny(eye_region, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    return bool(edge_density > 0.08)

def describe_face(bgr_crop):
    attrs = deepface_attrs(bgr_crop)
    attrs["glasses"] = detect_glasses_simple(bgr_crop)
    return attrs

if __name__ == "__main__":
    img = cv2.imread("../database/" + __import__("os").listdir("../database")[0] + "/00.jpg")
    print(describe_face(img))