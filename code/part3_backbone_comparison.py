import os, cv2, pickle
import numpy as np
import pandas as pd
from deepface import DeepFace
from detect_faces import detect_faces
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

DATABASE_DIR = "../database"
OUTPUT_CSV = "../outputs/backbone_comparison.csv"

def embed_face(bgr_crop, model_name):
    rep = DeepFace.represent(
        bgr_crop,
        model_name=model_name,
        enforce_detection=False,
        detector_backend="skip"
    )
    v = np.array(rep[0]["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)

def build_db_for_backbone(model_name):
    db = {}

    for name in sorted(os.listdir(DATABASE_DIR)):
        person_dir = os.path.join(DATABASE_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        embs = []
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue

            faces = detect_faces(img, conf=0.4)
            if not faces:
                continue

            x1, y1, x2, y2, _ = max(
                faces,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
            )

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            embs.append(embed_face(crop, model_name))

        if embs:
            db[name] = np.stack(embs)
            print(f"{model_name} | {name}: {len(embs)} embeddings")

    return db

def make_xy(db):
    X, y = [], []
    for name, embs in db.items():
        for e in embs:
            X.append(e)
            y.append(name)
    return np.array(X), np.array(y)

def evaluate(db):
    X, y = make_xy(db)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    models = {
        "1-NN cosine": KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        "3-NN cosine": KNeighborsClassifier(n_neighbors=3, metric="cosine"),
        "Linear SVM": SVC(kernel="linear", probability=True),
        "RBF SVM": SVC(kernel="rbf", C=10, probability=True),
    }

    accs = {}
    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        accs[name] = clf.score(X_te, y_te)

    return accs

if __name__ == "__main__":
    backbones = ["Facenet512", "ArcFace", "VGG-Face"]
    results = {}

    for backbone in backbones:
        print(f"\nRunning backbone: {backbone}")
        db = build_db_for_backbone(backbone)
        results[backbone] = evaluate(db)

    table = pd.DataFrame(results).T
    print("\nBackbone Comparison:")
    print(table)

    table.to_csv(OUTPUT_CSV)
    print(f"\nSaved: {OUTPUT_CSV}")