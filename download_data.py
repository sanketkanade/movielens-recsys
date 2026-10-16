"""Download MovieLens-1M from GroupLens into ./data (not committed to git).

The licence for MovieLens forbids redistributing the data, so the repo only
ships this script. We verify the archive against the MD5 that GroupLens
publishes, so every run (laptop or Docker) trains on byte-identical data.
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MD5 = "c4d9eecfca2ab87c1945afe126590906"  # from ml-1m.zip.md5 on grouplens.org
NEEDED = ["ratings.dat", "movies.dat", "users.dat"]


def download(data_dir: Path = Path("data")) -> Path:
    out_dir = data_dir / "ml-1m"
    if all((out_dir / f).exists() for f in NEEDED):
        print(f"[data] already present in {out_dir}")
        return out_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-1m.zip"
    print(f"[data] downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)

    digest = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if digest != MD5:
        sys.exit(f"[data] checksum mismatch: got {digest}, expected {MD5}")

    # Extract only the files we use, by exact name, rather than trusting
    # whatever paths the archive contains.
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in NEEDED:
            (out_dir / name).write_bytes(zf.read(f"ml-1m/{name}"))
    zip_path.unlink()
    print(f"[data] extracted {', '.join(NEEDED)} to {out_dir}")
    return out_dir


if __name__ == "__main__":
    download()
