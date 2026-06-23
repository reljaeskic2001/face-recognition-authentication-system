import cv2
import pickle
import time
import numpy as np

from detect_faces import detect_faces
from embeddings import embed_face
from identify import identify
from attributes import describe_face

DB_PATH = "../face_db.pkl"
ME = "relja"
THRESHOLD = 0.55
PANEL_W = 320

db = pickle.load(open(DB_PATH, "rb"))

def draw_panel(panel, status, fields, ok):
    panel[:] = (32, 32, 32)
    color = (0, 220, 0) if ok else (0, 0, 255)

    cv2.putText(panel, status, (16, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    y = 110
    for k, v in fields.items():
        cv2.putText(panel, f"{k}:", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(panel, str(v), (140, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 36

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

last_attrs = {
    "age": "-",
    "gender": "-",
    "expression": "-",
    "glasses": "-"
}

last_name = "-"
last_score = 0.0
last_ok = False
frame_id = 0
t0 = time.time()

while True:
    ok_cap, frame = cap.read()
    if not ok_cap:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]
    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)

    boxes = detect_faces(frame, conf=0.4)

    if boxes:
        x1, y1, x2, y2, _ = max(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
        )

        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            emb = embed_face(crop)
            last_name, last_score = identify(emb, db, threshold=THRESHOLD)
            last_ok = last_name == ME

            if frame_id % 10 == 0:
                last_attrs = describe_face(crop)

            color = (0, 220, 0) if last_ok else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    fps = (frame_id + 1) / (time.time() - t0 + 1e-6)
    status = "AUTHENTICATED" if last_ok else "DENIED"

    fields = {
        "Identity": f"{last_name} ({last_score:.2f})",
        "Age": last_attrs["age"],
        "Gender": last_attrs["gender"],
        "Expression": last_attrs["expression"],
        "Glasses": "Yes" if last_attrs["glasses"] else "No",
        "FPS": f"{fps:.1f}"
    }

    draw_panel(panel, status, fields, last_ok)

    output = np.hstack([frame, panel])
    cv2.imshow("Face Authentication - ESC to quit", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()