"""
Server (Railway) se dataset.zip download karke local dataset/ folder mein extract karta hai.

Chalane ka tareeqa:
    set SYNC_KEY=<Railway Variables wali SYNC_KEY>      (Windows)
    export SYNC_KEY=<...>                                (Linux/Mac)
    python sync_dataset.py https://your-app.up.railway.app

Key ko kabhi code ya git mein mat likhein.
"""
import io
import os
import sys
import urllib.parse
import urllib.request
import zipfile

base = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SYNC_URL", "")).rstrip("/")
key = os.environ.get("SYNC_KEY", "")
if not base or not key:
    sys.exit("Usage: SYNC_KEY=... python sync_dataset.py https://your-app.up.railway.app")

url = f"{base}/api/dataset_zip?key={urllib.parse.quote(key)}"
with urllib.request.urlopen(url, timeout=120) as r:
    data = r.read()

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(dest, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as zf:
    zf.extractall(dest)
    print(f"Done! {len(zf.namelist())} files '{dest}' mein aa gayi (images + manifest.csv).")
