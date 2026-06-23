import os
import pickle
import cv2
import numpy as np
from deepface import DeepFace
from detect_faces import detect_faces

EMBED_MODEL = "Facenet512"
DATABASE_DIR = "../database"
OUTPUT_PATH = "../face_db.pkl"

def embed_face(bgr_crop):
    rep = DeepFace.represent(
        bgr_crop,
        model_name=EMBED_MODEL,
        enforce_detection=False,
        detector_backend="skip"
    )

    v = np.array(rep[0]["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)

def build_database():
    db = {}

    for name in sorted(os.listdir(DATABASE_DIR)):
        person_dir = os.path.join(DATABASE_DIR, name)

        if not os.path.isdir(person_dir):
            continue

        embs = []

        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)

            if img is None:
                continue

            faces = detect_faces(img, conf=0.4)

            if not faces:
                print(f"No face found: {img_path}")
                continue

            x1, y1, x2, y2, _ = max(
                faces,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
            )

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            emb = embed_face(crop)
            embs.append(emb)

        if embs:
            db[name] = np.stack(embs)
            print(f"{name}: {len(embs)} embeddings")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(db, f)

    total_embeddings = sum(len(v) for v in db.values())
    print("\nDONE")
    print(f"Identities: {len(db)}")
    print(f"Total embeddings: {total_embeddings}")
    print(f"Saved: {OUTPUT_PATH}")

    return db

if __name__ == "__main__":
    build_database()