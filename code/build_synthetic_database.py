import os, time, uuid, requests, cv2, numpy as np

URL = "https://thispersondoesnotexist.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}
NAMES = ["ava", "ben", "cara", "dev", "eli", "finn", "gia", "hugo", "iris", "jade", "kai", "lia"]

ROOT = "../database"

def random_id(i):
    return f"{NAMES[i % len(NAMES)]}_{uuid.uuid4().hex[:6]}"

def fetch_face():
    r = requests.get(URL, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.content

def make_views(img):
    h, w = img.shape[:2]
    flipped = cv2.flip(img, 1)
    darker = np.clip(img.astype(np.int16) - 25, 0, 255).astype(np.uint8)
    brighter = np.clip(img.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    cy, cx, s = h // 2, w // 2, int(min(h, w) * 0.42)
    zoomed = cv2.resize(img[cy-s:cy+s, cx-s:cx+s], (w, h))
    return [img, flipped, darker, brighter, zoomed]

def download_synthetic_database(n_identities=12):
    os.makedirs(ROOT, exist_ok=True)

    for i in range(n_identities):
        pid = random_id(i)
        pdir = os.path.join(ROOT, pid)
        os.makedirs(pdir, exist_ok=True)

        raw = fetch_face()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

        for j, view in enumerate(make_views(img)):
            cv2.imwrite(os.path.join(pdir, f"{j:02d}.jpg"), view)

        print(f"[{i+1}/{n_identities}] saved {pid}")
        time.sleep(1.5)

if __name__ == "__main__":
    download_synthetic_database(12)