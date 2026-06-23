import pickle
import numpy as np

db = pickle.load(open("../face_db.pkl", "rb"))

def identify(query_emb, db, threshold=0.55):
    best_name = "Unknown"
    best_score = -1.0

    for name, embs in db.items():
        sims = embs @ query_emb
        score = float(sims.max())

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return "Unknown", best_score

    return best_name, best_score