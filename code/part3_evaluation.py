import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

DB_PATH = "../face_db.pkl"
OUTPUT_DIR = "../outputs"

def make_xy(db):
    X, y = [], []

    for name, embs in db.items():
        for e in embs:
            X.append(e)
            y.append(name)

    return np.array(X), np.array(y)

def evaluate_classifiers(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=0
    )

    models = {
        "1-NN cosine": KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        "3-NN cosine": KNeighborsClassifier(n_neighbors=3, metric="cosine"),
        "Linear SVM": SVC(kernel="linear", probability=True),
        "RBF SVM": SVC(kernel="rbf", C=10, probability=True)
    }

    print("\nClassifier Results:")
    print("-------------------")

    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        print(f"{name}: {acc:.3f}")

def make_similarity_heatmap(db):
    names = sorted(db.keys())

    # use first embedding as held-out query, rest as database
    query_embs = []
    db_embs = {}

    for name in names:
        query_embs.append(db[name][0])
        db_embs[name] = db[name][1:]

    sim_matrix = []

    correct = 0

    for i, query in enumerate(query_embs):
        row = []

        for name in names:
            sims = db_embs[name] @ query
            row.append(float(sims.max()))

        row = np.array(row)
        sim_matrix.append(row)

        predicted = names[np.argmax(row)]
        if predicted == names[i]:
            correct += 1

    sim_matrix = np.array(sim_matrix)
    top1_acc = correct / len(names)

    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix)
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.title(f"Similarity Heatmap, Top-1 Accuracy = {top1_acc:.3f}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/similarity_heatmap.png")
    plt.close()

    print(f"\nTop-1 cosine accuracy: {top1_acc:.3f}")
    print(f"Saved heatmap: {OUTPUT_DIR}/similarity_heatmap.png")

if __name__ == "__main__":
    db = pickle.load(open(DB_PATH, "rb"))

    print(f"Identities: {len(db)}")
    print(f"Total embeddings: {sum(len(v) for v in db.values())}")

    X, y = make_xy(db)
    evaluate_classifiers(X, y)
    make_similarity_heatmap(db)