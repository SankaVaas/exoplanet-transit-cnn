"""
Diagnostic script: isolate whether MAST download failures are specific to
the mast.stsci.edu API domain (as opposed to lightkurve/astroquery's code)
by finding the actual download URI lightkurve would use, then fetching it
with plain `requests` directly — bypassing lightkurve/astroquery's
download machinery entirely.

Run from the project root with the venv active:
    python diagnose_mast.py
"""
import time
import requests
import lightkurve as lk

KEPID = 1432214  # one of the stars that failed in the real run

print(f"Searching for KIC {KEPID}...")
search_result = lk.search_lightcurve(f"KIC {KEPID}", mission="Kepler", author="Kepler", cadence="long")
print(f"Found {len(search_result)} products.\n")

if len(search_result) == 0:
    print("No products found — nothing to test.")
    raise SystemExit(0)

table = search_result.table
print("Available columns:", list(table.colnames))
print()

uri_col = None
for candidate in ["dataURI", "dataURL", "productFilename"]:
    if candidate in table.colnames:
        uri_col = candidate
        break

if uri_col is None:
    print("Could not find a URI column automatically. Dumping first row for inspection:")
    print(table[0])
    raise SystemExit(0)

data_uri = table[uri_col][0]
print(f"Using column '{uri_col}': {data_uri}")

# Build the standard MAST download-by-URI API endpoint.
download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}"
print(f"\nAttempting direct requests.get() against:\n  {download_url}\n")

t0 = time.time()
try:
    with requests.get(download_url, stream=True, timeout=120) as resp:
        print("HTTP status:", resp.status_code)
        print("Headers:", dict(resp.headers))
        expected_size = int(resp.headers.get("Content-Length", -1))
        print(f"Content-Length header: {expected_size} bytes")

        total = 0
        with open("mast_test_download.fits", "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                total += len(chunk)

        elapsed = time.time() - t0
        print(f"\nDownloaded {total} bytes in {elapsed:.1f}s")
        if expected_size > 0:
            print(f"Match expected size: {total == expected_size}")
except requests.exceptions.RequestException as e:
    print(f"\nDirect request FAILED: {type(e).__name__}: {e}")