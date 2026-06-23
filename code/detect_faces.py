from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "../yolov8n-face.pt"
face_detector = YOLO(MODEL_PATH)

def detect_faces(bgr_image, conf=0.4):
    results = face_detector.predict(bgr_image, conf=conf, verbose=False)
    boxes = []

    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            boxes.append((x1, y1, x2, y2, float(b.conf[0])))

    return boxes

if __name__ == "__main__":
    person = os.listdir("../database")[0]
    img_path = f"../database/{person}/00.jpg"

    img = cv2.imread(img_path)
    boxes = detect_faces(img)

    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{c:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    os.makedirs("../outputs", exist_ok=True)
    cv2.imwrite("../outputs/detected_test.jpg", img)
    print("Saved: ../outputs/detected_test.jpg")