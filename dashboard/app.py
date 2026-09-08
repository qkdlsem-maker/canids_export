import json
from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "lgbm_v2.txt"
MAHA_PATH = ROOT / "models" / "maha_v2.npz"
STATS_PATH = ROOT / "models" / "idagnostic_stats.pkl"

FEATURE_NAMES = [
    "freq_in_window",
    "unique_ids_in_window",
    "entropy",
    "mean_byte",
    "global_delta_zscore",
    "global_value_zscore",
]

CLASS_NAMES = ["DoS", "Fuzzy", "R", "RPM", "gear"]
NORMAL_IDX = 2
CODE_MAP = [1, 2, 0, 3, 4]

app = Flask(__name__)

from v2_stream import V2StreamFeatureExtractor

stream = V2StreamFeatureExtractor(STATS_PATH)

print("[JARVIS] Loading frozen V2 model...")

lgbm = lgb.Booster(model_file=str(MODEL_PATH))

with np.load(MAHA_PATH) as z:
    maha_mean = np.asarray(z["mean"], dtype=np.float64)
    maha_inv_cov = np.asarray(z["inv_cov"], dtype=np.float64)
    maha_threshold = float(z["thr"])

with open(STATS_PATH, "rb") as f:
    idagnostic_stats = pickle.load(f)

print("[JARVIS] Frozen V2 loaded.")
print(f"[JARVIS] LightGBM features : {lgbm.num_feature()}")
print(f"[JARVIS] Mahalanobis dim   : {maha_mean.shape}")
print(f"[JARVIS] Threshold         : {maha_threshold:.12f}")


def mahalanobis_distance(features):
    x = np.asarray(features, dtype=np.float64)
    d = x - maha_mean
    return float(np.sqrt(d @ maha_inv_cov @ d))


def hybrid_predict(features):
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)

    raw = np.asarray(lgbm.predict(x)[0], dtype=np.float64)
    best_idx = int(np.argmax(raw))

    distance = mahalanobis_distance(x[0])

    if best_idx != NORMAL_IDX:
        final_code = CODE_MAP[best_idx]
        stage = "LightGBM"
    elif distance > maha_threshold:
        final_code = 5
        stage = "Mahalanobis"
    else:
        final_code = 0
        stage = "Normal"

    return {
        "lightgbm_class": CLASS_NAMES[best_idx],
        "lightgbm_index": best_idx,
        "scores": raw.tolist(),
        "mahalanobis_distance": distance,
        "mahalanobis_threshold": maha_threshold,
        "final_code": final_code,
        "decision_stage": stage,
    }


DEMO_BUNDLE_PATH = ROOT / "dashboard" / "data" / "demo_bundle.json"

with DEMO_BUNDLE_PATH.open(encoding="utf-8") as f:
    DEMO_BUNDLE = json.load(f)


@app.get("/api/scenarios")
def scenarios():
    items = []

    for key, value in DEMO_BUNDLE["scenarios"].items():
        items.append({
            "id": key,
            "display_name": value["display_name"],
            "source": value["source"],
            "frame_count": len(value["frames"]),
            "note": value.get("note"),
            "warmup_frames": value.get("warmup_frames", 0),
        })

    return jsonify({
        "status": "ok",
        "scenarios": items,
    })


@app.get("/api/scenario/<name>")
def scenario(name):
    scenarios = DEMO_BUNDLE["scenarios"]

    if name not in scenarios:
        return jsonify({
            "status": "error",
            "message": f"unknown scenario: {name}",
        }), 404

    value = scenarios[name]

    return jsonify({
        "status": "ok",
        "scenario": {
            "id": name,
            "display_name": value["display_name"],
            "source": value["source"],
            "note": value.get("note"),
            "warmup_frames": value.get("warmup_frames", 0),
            "injection_id": value.get("injection_id"),
            "injection_payload": value.get("injection_payload"),
            "injection_interval_sec": value.get(
                "injection_interval_sec"
            ),
            "frames": value["frames"],
        },
    })


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/reset")
def reset_stream():
    stream.reset()
    return jsonify({
        "status": "ok",
        "message": "V2 stream state reset",
    })


@app.post("/api/frame")
def process_frame():
    body = request.get_json(silent=True) or {}

    if "can_id" not in body or "payload" not in body:
        return jsonify({
            "status": "error",
            "message": "can_id and payload are required",
        }), 400

    try:
        can_id = body["can_id"]

        if isinstance(can_id, str):
            can_id = int(can_id, 16)
        else:
            can_id = int(can_id)

        result = stream.process(can_id, body["payload"])
        prediction = hybrid_predict(result["features"])

        return jsonify({
            "status": "ok",
            "frame": {
                "can_id": result["can_id"],
                "can_id_hex": result["can_id_hex"],
                "payload_hex": result["payload_hex"],
            },
            "features": result["feature_dict"],
            "prediction": prediction,
        })

    except (ValueError, TypeError) as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "system": "J.A.R.V.I.S CAN IDS",
        "model": "Frozen V2 Hybrid",
        "features": FEATURE_NAMES,
        "num_features": lgbm.num_feature(),
        "mahalanobis_threshold": maha_threshold,
    })


if __name__ == "__main__":
    # Local-only judging demo. No Internet connection required.
    app.run(host="127.0.0.1", port=5000, debug=False)
