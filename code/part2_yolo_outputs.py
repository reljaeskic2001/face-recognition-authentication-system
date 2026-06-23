from ultralytics import YOLO
import cv2
import os
import glob

MODEL_PATH = "../yolov8n-face.pt"
DATABASE_DIR = "../database"
OUTPUT_DIR = "../outputs/part2"

face_detector = YOLO(MODEL_PATH)

def detect_faces(bgr_image, conf=0.4):
    results = face_detector.predict(bgr_image, conf=conf, verbose=False)
    boxes = []

    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            boxes.append((x1, y1, x2, y2, float(b.conf[0])))

    return boxes

def draw_boxes(img, boxes):
    out = img.copy()

    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{c:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return out

def run_three_database_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = []
    people = sorted(os.listdir(DATABASE_DIR))

    for person in people:
        person_dir = os.path.join(DATABASE_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        imgs = glob.glob(os.path.join(person_dir, "*.jpg"))
        if imgs:
            image_paths.append(imgs[0])

        if len(image_paths) == 3:
            break

    for i, path in enumerate(image_paths, start=1):
        img = cv2.imread(path)
        boxes = detect_faces(img)
        out = draw_boxes(img, boxes)

        save_path = os.path.join(OUTPUT_DIR, f"database_detection_{i}.jpg")
        cv2.imwrite(save_path, out)

        print(f"Database image {i}: detected {len(boxes)} face(s)")
        print(f"Saved: {save_path}")

def run_group_photo():
    group_path = "../group_photo.jpg"

    if not os.path.exists(group_path):
        print("\nNo group_photo.jpg found.")
        print("Put a group photo with 5+ faces in the main project folder.")
        print("Name it exactly: group_photo.jpg")
        return

    img = cv2.imread(group_path)
    boxes = detect_faces(img)
    out = draw_boxes(img, boxes)

    save_path = os.path.join(OUTPUT_DIR, "group_photo_detection.jpg")
    cv2.imwrite(save_path, out)

    print(f"\nGroup photo: detected {len(boxes)} face(s)")
    print(f"Saved: {save_path}")
    print("Now manually count the true number of faces in the original group photo.")

if __name__ == "__main__":
    run_three_database_images()
    run_group_photo()