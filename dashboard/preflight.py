from pathlib import Path
import hashlib
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    "LightGBM model": ROOT / "models" / "lgbm_v2.txt",
    "Mahalanobis model": ROOT / "models" / "maha_v2.npz",
    "V2 statistics": ROOT / "models" / "idagnostic_stats.pkl",
    "Demo bundle": ROOT / "dashboard" / "data" / "demo_bundle.json",
    "Dashboard HTML": ROOT / "dashboard" / "templates" / "index.html",
    "Dashboard CSS": ROOT / "dashboard" / "static" / "style.css",
    "Dashboard JS": ROOT / "dashboard" / "static" / "app.js",
    "V2 stream extractor": ROOT / "dashboard" / "v2_stream.py",
}

# Frozen inference assets only.
# UI files are intentionally not hash-locked because they may still be tuned.
EXPECTED_SHA256 = {
    ROOT / "models" / "idagnostic_stats.pkl":
        "04f3af021d6c9e3d52e6060c93d8fe14fd2a381242d8bbbeb93a0aee11d0f37b",
    ROOT / "dashboard" / "data" / "demo_bundle.json":
        "1a2d33485e286ede1d6961d24b65d80ba8365e17b5a44b8be8beda7a4f632cdf",
}

EXPECTED_SCENARIOS = {
    "normal": 300,
    "dos": 300,
    "fuzzy": 300,
    "spoof": 300,
    "zero_day": 400,
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


print("=" * 62)
print(" J.A.R.V.I.S CAN IDS - OFFLINE DEMO PREFLIGHT")
print("=" * 62)

print(f"[INFO] Python: {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    fail("Python 3.10 or newer is required.")

print("\n[1/5] Python packages")
try:
    import flask
    import numpy as np
    import lightgbm as lgb
except ImportError as e:
    fail(f"Missing Python package: {e}")

print(f"[PASS] Flask    {flask.__version__}")
print(f"[PASS] NumPy    {np.__version__}")
print(f"[PASS] LightGBM {lgb.__version__}")

print("\n[2/5] Required files")
for label, path in REQUIRED_FILES.items():
    if not path.is_file():
        fail(f"{label} missing: {path}")
    print(f"[PASS] {label}")

print("\n[3/5] Frozen asset integrity")
for path, expected in EXPECTED_SHA256.items():
    actual = sha256(path)
    if actual != expected:
        fail(
            f"SHA256 mismatch: {path.name}\n"
            f"       expected {expected}\n"
            f"       actual   {actual}"
        )
    print(f"[PASS] {path.name} SHA256")

print("\n[4/5] Frozen V2 model validation")

model_path = ROOT / "models" / "lgbm_v2.txt"
maha_path = ROOT / "models" / "maha_v2.npz"
stats_path = ROOT / "models" / "idagnostic_stats.pkl"

booster = lgb.Booster(model_file=str(model_path))
if booster.num_feature() != 6:
    fail(f"LightGBM feature count is {booster.num_feature()}, expected 6.")
print("[PASS] LightGBM num_feature = 6")

with np.load(maha_path) as z:
    required_keys = {"mean", "inv_cov", "thr"}
    if not required_keys.issubset(z.files):
        fail(f"Mahalanobis NPZ keys invalid: {z.files}")

    mean = np.asarray(z["mean"])
    inv_cov = np.asarray(z["inv_cov"])
    threshold = float(z["thr"])

if mean.shape != (6,):
    fail(f"Mahalanobis mean shape {mean.shape}, expected (6,)")
if inv_cov.shape != (6, 6):
    fail(f"Mahalanobis inv_cov shape {inv_cov.shape}, expected (6, 6)")
if abs(threshold - 10.104021265036314) > 1e-9:
    fail(f"Unexpected Mahalanobis threshold: {threshold}")

print("[PASS] Mahalanobis dimension = 6")
print(f"[PASS] Threshold = {threshold:.12f}")

with stats_path.open("rb") as f:
    stats = pickle.load(f)

if int(stats.get("window", -1)) != 20:
    fail(f"Unexpected feature window: {stats.get('window')}")

print("[PASS] Streaming feature window = 20")

print("\n[5/5] Demo bundle")
bundle_path = ROOT / "dashboard" / "data" / "demo_bundle.json"

with bundle_path.open("r", encoding="utf-8") as f:
    bundle = json.load(f)

scenarios = bundle.get("scenarios")
if not isinstance(scenarios, dict):
    fail("demo_bundle.json has no scenarios object.")

for name, expected_count in EXPECTED_SCENARIOS.items():
    if name not in scenarios:
        fail(f"Scenario missing: {name}")

    frames = scenarios[name].get("frames", [])
    if len(frames) != expected_count:
        fail(
            f"{name}: {len(frames)} frames, "
            f"expected {expected_count}"
        )

    print(f"[PASS] {name:8s}: {len(frames)} frames")

print("\n" + "=" * 62)
print(" PREFLIGHT PASS - OFFLINE DEMO READY")
print("=" * 62)
