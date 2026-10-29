"""Download the public tutoring logs (ASSISTments and Cognitive Tutor) used in the paper.

Fetches:
- ct.csv (~4.3 MB): Cognitive Tutor algebra log
- as.csv (~83 MB): ASSISTments 2009-2010 skill builder log
"""
import os
import shutil
import subprocess
import urllib.request
import ssl

DATA_DIR = "data"
URLS = {
    "ct.csv": "https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/ct.csv",
    "as.csv": "https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv",
}


def download_file(url, dest):
    print(f"Downloading {url} -> {dest}...")
    if shutil.which("curl"):
        res = subprocess.run(["curl", "-L", "-o", dest, url])
        if res.returncode == 0:
            return
    try:
        ctx = ssl.create_default_context()
        urllib.request.urlretrieve(url, dest)
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ctx) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname, url in URLS.items():
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"{dest} already exists ({os.path.getsize(dest)} bytes), skipping.")
            continue
        download_file(url, dest)
        print(f"Saved {dest} ({os.path.getsize(dest)} bytes).")
    print("All datasets ready.")


if __name__ == "__main__":
    main()
