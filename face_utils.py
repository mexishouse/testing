"""
face_utils.py
OpenCV face detection, sample storage, LBPH training and 1:1 verification.

Dataset layout (everything is also packed into a single ZIP automatically):
    dataset/<employee_id>/<n>.jpg          -> raw face samples
    dataset_archive/dataset.zip            -> <employee_id>/<n>.jpg + manifest.csv
"""
import csv
import io
import os
import shutil
import tempfile
import threading
import zipfile

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
ARCHIVE_DIR = os.path.join(DATA_DIR, "dataset_archive")
ZIP_PATH = os.path.join(ARCHIVE_DIR, "dataset.zip")
TRAINER_PATH = os.path.join(DATA_DIR, "trainer", "trainer.yml")

for _d in (DATASET_DIR, ARCHIVE_DIR, os.path.dirname(TRAINER_PATH)):
    os.makedirs(_d, exist_ok=True)

_LOCAL_CASCADE = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
FACE_CASCADE = cv2.CascadeClassifier(_LOCAL_CASCADE)
if FACE_CASCADE.empty() and hasattr(cv2, "data"):
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if FACE_CASCADE.empty():
    raise RuntimeError("Could not load haarcascade_frontalface_default.xml")

SAMPLES_NEEDED = int(os.environ.get("SAMPLES_NEEDED", 50))
FACE_SIZE = (200, 200)
# LBPH: LOWER distance = better match.  Lower this number to be stricter.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 75))
DUPLICATE_THRESHOLD = 65          # used to warn the admin when a new face looks like an existing employee
STREAK_NEEDED = int(os.environ.get("STREAK_NEEDED", 3))   # consecutive matching frames required

_lock = threading.RLock()
_recognizer = None
_recognizer_mtime = None


# ------------------------------------------------------------------ image helpers
def preprocess_face(gray_face):
    return cv2.equalizeHist(cv2.resize(gray_face, FACE_SIZE))


def detect_faces(gray):
    return FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))


def largest_face(faces):
    return max(faces, key=lambda f: f[2] * f[3])


def decode_image(data_url, max_width=640):
    """base64 data-URL -> BGR image (resized) or None."""
    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        arr = np.frombuffer(__import__("base64").b64decode(encoded), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
        return frame
    except Exception:
        return None


# ------------------------------------------------------------------ dataset
def employee_dir(emp_id):
    path = os.path.join(DATASET_DIR, str(emp_id))
    os.makedirs(path, exist_ok=True)
    return path


def count_samples(emp_id):
    path = os.path.join(DATASET_DIR, str(emp_id))
    if not os.path.isdir(path):
        return 0
    return len([f for f in os.listdir(path) if f.lower().endswith(".jpg")])


def reset_samples(emp_id):
    path = os.path.join(DATASET_DIR, str(emp_id))
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def save_face_sample(emp_id, gray_face):
    """Saves the next numbered sample and returns the new count."""
    n = count_samples(emp_id) + 1
    cv2.imwrite(os.path.join(employee_dir(emp_id), f"{n}.jpg"), cv2.resize(gray_face, FACE_SIZE))
    return n


def sample_path(emp_id, n):
    p = os.path.join(DATASET_DIR, str(emp_id), f"{n}.jpg")
    return p if os.path.isfile(p) else None


def rebuild_dataset_zip(employees=None):
    """Packs the whole dataset into dataset_archive/dataset.zip (atomic replace).
    `employees` = optional list of dicts (id, full_name, username, department_name, status)
    used to write manifest.csv next to the images."""
    with _lock:
        fd, tmp = tempfile.mkstemp(suffix=".zip", dir=ARCHIVE_DIR)
        os.close(fd)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for emp_id in sorted(os.listdir(DATASET_DIR)):
                d = os.path.join(DATASET_DIR, emp_id)
                if not os.path.isdir(d):
                    continue
                for img in sorted(os.listdir(d)):
                    zf.write(os.path.join(d, img), arcname=f"{emp_id}/{img}")
            if employees:
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["employee_id", "full_name", "username", "department", "status", "samples"])
                for e in employees:
                    w.writerow([e["id"], e["full_name"], e["username"], e["department_name"],
                                e["status"], count_samples(e["id"])])
                zf.writestr("manifest.csv", buf.getvalue())
        os.replace(tmp, ZIP_PATH)
    return ZIP_PATH


# ------------------------------------------------------------------ training / recognizer
def train_model():
    """Train LBPH on every dataset/<employee_id>/*.jpg. Returns (ok, message)."""
    global _recognizer, _recognizer_mtime
    if not hasattr(cv2, "face"):
        return False, ("cv2.face not found. Install opencv-contrib-python-headless "
                       "(and uninstall opencv-python).")
    faces, labels = [], []
    with _lock:
        for name in os.listdir(DATASET_DIR):
            d = os.path.join(DATASET_DIR, name)
            if not (os.path.isdir(d) and name.isdigit()):
                continue
            for img_name in os.listdir(d):
                img = cv2.imread(os.path.join(d, img_name), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(cv2.equalizeHist(cv2.resize(img, FACE_SIZE)))
                labels.append(int(name))
        if not faces:
            if os.path.exists(TRAINER_PATH):
                os.remove(TRAINER_PATH)
            _recognizer, _recognizer_mtime = None, None
            return True, "No samples left; model cleared."
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.train(faces, np.array(labels))
        rec.save(TRAINER_PATH)
        _recognizer, _recognizer_mtime = rec, os.path.getmtime(TRAINER_PATH)
    return True, f"Model trained on {len(faces)} samples."


def get_recognizer():
    """Cached recognizer; reloads automatically when trainer.yml changes."""
    global _recognizer, _recognizer_mtime
    with _lock:
        if not os.path.exists(TRAINER_PATH) or not hasattr(cv2, "face"):
            return None
        mtime = os.path.getmtime(TRAINER_PATH)
        if _recognizer is None or mtime != _recognizer_mtime:
            rec = cv2.face.LBPHFaceRecognizer_create()
            rec.read(TRAINER_PATH)
            _recognizer, _recognizer_mtime = rec, mtime
        return _recognizer


def delete_employee_data(emp_id, employees=None):
    reset_samples(emp_id)
    train_model()
    rebuild_dataset_zip(employees)


# ------------------------------------------------------------------ verification
def verify_face(gray, employee_id):
    """1:1 check — is the face in front of the camera THIS employee?
    Returns dict(state, match, confidence, box)
    state: no_face | multiple_faces | no_model | mismatch | match"""
    faces = detect_faces(gray)
    if len(faces) == 0:
        return {"state": "no_face", "match": False}
    x, y, w, h = [int(v) for v in largest_face(faces)]
    others = [f for f in faces if f[2] * f[3] > 0.5 * w * h and tuple(int(v) for v in f) != (x, y, w, h)]
    if others:
        return {"state": "multiple_faces", "match": False, "box": [x, y, w, h]}
    rec = get_recognizer()
    if rec is None:
        return {"state": "no_model", "match": False, "box": [x, y, w, h]}
    with _lock:
        label, conf = rec.predict(preprocess_face(gray[y:y + h, x:x + w]))
    ok = (label == employee_id) and conf <= CONFIDENCE_THRESHOLD
    return {"state": "match" if ok else "mismatch", "match": ok,
            "confidence": round(float(conf), 1), "box": [x, y, w, h]}


def find_similar_existing(emp_id):
    """Run an employee's fresh samples against the model trained BEFORE them.
    If they strongly resemble somebody who is already enrolled, return (other_id, ratio)."""
    rec = get_recognizer()
    d = os.path.join(DATASET_DIR, str(emp_id))
    if rec is None or not os.path.isdir(d):
        return None
    files = sorted(os.listdir(d))[::3] or []
    hits, total = {}, 0
    for f in files:
        img = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        with _lock:
            label, conf = rec.predict(cv2.equalizeHist(cv2.resize(img, FACE_SIZE)))
        total += 1
        if label != emp_id and conf <= DUPLICATE_THRESHOLD:
            hits[label] = hits.get(label, 0) + 1
    if not total or not hits:
        return None
    other, n = max(hits.items(), key=lambda kv: kv[1])
    return (other, n / total) if n / total >= 0.4 else None
