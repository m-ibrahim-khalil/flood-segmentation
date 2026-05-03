"""Download Sen1Floods11 HandLabeled chips from GCS without gsutil/gcloud.

The bucket gs://sen1floods11 is public; we use the unauthenticated JSON
list-objects API (storage.googleapis.com/storage/v1/b/...) and HTTPS GETs.
This avoids stale gsutil auth (`invalid_grant`) and matches what onnxruntime
or any cloud-agnostic CI would do.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
import json
import ssl
import sys
import time

import urllib.request


# python.org's Python on macOS does not see the system cert bundle by default,
# producing CERTIFICATE_VERIFY_FAILED on the very first HTTPS call. Build an
# SSL context that prefers certifi's bundle when available, then falls back
# to the OS truststore.
def _make_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL_CTX = _make_ssl_context()
_HTTPS_HANDLER = urllib.request.HTTPSHandler(context=_SSL_CTX)
_OPENER = urllib.request.build_opener(_HTTPS_HANDLER)


BASE = "https://storage.googleapis.com"
BUCKET = "sen1floods11"


def list_objects(prefix: str):
    """Yield (object_name, size_bytes) for every object whose name starts with prefix."""
    page_token = None
    while True:
        url = (f"{BASE}/storage/v1/b/{BUCKET}/o"
               f"?prefix={quote(prefix)}&maxResults=1000")
        if page_token:
            url += f"&pageToken={quote(page_token)}"
        with _OPENER.open(url) as r:
            data = json.loads(r.read().decode())
        for it in data.get("items", []):
            yield it["name"], int(it.get("size", 0))
        page_token = data.get("nextPageToken")
        if not page_token:
            return


def download_one(name: str, dest_root: Path) -> tuple[str, bool, str]:
    out = dest_root / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return name, True, "skip-exists"
    url = f"{BASE}/{BUCKET}/{quote(name, safe='/')}"
    try:
        with _OPENER.open(url) as r, open(out, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        return name, True, "ok"
    except Exception as e:
        return name, False, f"err:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sen1floods11_raw",
                    help="Local output directory (mirrors bucket layout under it).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many files per prefix (0 = all). Use --limit 300 "
                         "to grab ~250 chips + 50 spare for sampling.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefixes", nargs="+", default=[
        "v1.1/data/flood_events/HandLabeled/S2Hand/",
        "v1.1/data/flood_events/HandLabeled/LabelHand/",
    ])
    args = ap.parse_args()

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    for prefix in args.prefixes:
        print(f"\nListing {prefix} ...", flush=True)
        all_items = list(list_objects(prefix))
        if args.limit:
            all_items = all_items[:args.limit]
        total_bytes = sum(s for _, s in all_items)
        print(f"  {len(all_items)} files, ~{total_bytes/1e6:.1f} MB")

        t0 = time.time()
        ok = 0; fail = 0; skip = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(download_one, name, dest) for name, _ in all_items]
            for i, fut in enumerate(as_completed(futs), 1):
                name, success, msg = fut.result()
                if msg == "skip-exists":
                    skip += 1
                elif success:
                    ok += 1
                else:
                    fail += 1
                    print(f"  FAIL {name}: {msg}", file=sys.stderr)
                if i % 50 == 0 or i == len(futs):
                    print(f"  {i}/{len(futs)} done (ok={ok} skip={skip} fail={fail})", flush=True)
        print(f"  finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
