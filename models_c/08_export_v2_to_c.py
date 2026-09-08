"""
Export the already-frozen final V2 hybrid IDS to C.

IMPORTANT:
- Does NOT retrain LightGBM.
- Loads models/lgbm_v2.txt exactly as saved.
- Loads models/maha_v2.npz exactly as saved.
- Final V2 uses 6 ID-agnostic features.
"""

import os
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LGB_PATH = os.path.join(ROOT, "models", "lgbm_v2.txt")
MAHA_PATH = os.path.join(ROOT, "models", "maha_v2.npz")
OUT_PATH = os.path.join(HERE, "can_ids_embedded_v2.c")

FEATURES = [
    "freq_in_window",
    "unique_ids_in_window",
    "entropy",
    "mean_byte",
    "global_delta_zscore",
    "global_value_zscore",
]

# Same ordering used by src/14_train_v2_and_road_test.py:
# classes = sorted(train["Label"].unique())
CLASSES_SORTED = ["DoS", "Fuzzy", "R", "RPM", "gear"]

# Public return codes used by the embedded hybrid:
# 0=R, 1=DoS, 2=Fuzzy, 3=RPM, 4=gear, 5=unknown/anomaly
OUTPUT_ORDER = ["R", "DoS", "Fuzzy", "RPM", "gear"]
NORMAL_IDX = CLASSES_SORTED.index("R")
CODE_MAP = [OUTPUT_ORDER.index(c) for c in CLASSES_SORTED]

booster = lgb.Booster(model_file=LGB_PATH)
model = booster.dump_model()

if booster.num_feature() != 6:
    raise RuntimeError(
        f"Expected frozen V2 to have 6 features, got {booster.num_feature()}"
    )

if booster.feature_name() != FEATURES:
    raise RuntimeError(
        "Frozen V2 feature order mismatch:\n"
        f"expected={FEATURES}\n"
        f"actual={booster.feature_name()}"
    )

if model["num_class"] != 5:
    raise RuntimeError(f"Expected 5 classes, got {model['num_class']}")

if model["num_tree_per_iteration"] != 5:
    raise RuntimeError(
        f"Expected 5 trees/iteration, got {model['num_tree_per_iteration']}"
    )

trees = model["tree_info"]

if len(trees) != 400:
    raise RuntimeError(f"Expected 400 trees, got {len(trees)}")


def fmt(v):
    return format(float(v), ".17g")


def emit_node(node, indent="    "):
    if "leaf_value" in node:
        return f"{indent}return {fmt(node['leaf_value'])};\n"

    decision = node.get("decision_type")
    missing = node.get("missing_type")

    if decision != "<=":
        raise RuntimeError(
            f"Unsupported decision type: {decision}"
        )

    if missing != "None":
        raise RuntimeError(
            f"Unsupported missing type: {missing}"
        )

    feature = int(node["split_feature"])
    threshold = fmt(node["threshold"])

    s = f"{indent}if (x[{feature}] <= {threshold}) {{\n"
    s += emit_node(node["left_child"], indent + "    ")
    s += f"{indent}}} else {{\n"
    s += emit_node(node["right_child"], indent + "    ")
    s += f"{indent}}}\n"
    return s


parts = []

parts.append(
"""/*
 * Frozen final V2 CAN IDS model.
 *
 * Generated from:
 *   models/lgbm_v2.txt
 *   models/maha_v2.npz
 *
 * NO RETRAINING is performed by this exporter.
 *
 * V2 feature order:
 *   0 freq_in_window
 *   1 unique_ids_in_window
 *   2 entropy
 *   3 mean_byte
 *   4 global_delta_zscore
 *   5 global_value_zscore
 *
 * Return codes:
 *   0 normal (R)
 *   1 DoS
 *   2 Fuzzy
 *   3 RPM
 *   4 gear
 *   5 unknown / Mahalanobis anomaly
 */

#include <math.h>

#define CAN_IDS_NUM_FEATURES 6
#define CAN_IDS_NUM_CLASSES 5
#define NORMAL_IDX 2

static const int CODE_MAP[CAN_IDS_NUM_CLASSES] = {1, 2, 0, 3, 4};

"""
)

# Each LightGBM tree becomes one deterministic C function.
for tree_idx, tree in enumerate(trees):
    parts.append(
        f"static double tree_{tree_idx}(const double *x)\n{{\n"
    )
    parts.append(emit_node(tree["tree_structure"]))
    parts.append("}\n\n")


# score(): raw LightGBM multiclass scores.
#
# LightGBM tree order for multiclass:
# tree 0 -> class 0
# tree 1 -> class 1
# ...
# tree 4 -> class 4
# tree 5 -> class 0 of next boosting iteration, etc.
parts.append(
"""void score(const double *x, double *output)
{
    for (int c = 0; c < CAN_IDS_NUM_CLASSES; ++c) {
        output[c] = 0.0;
    }

"""
)

for tree_idx in range(len(trees)):
    class_idx = tree_idx % 5
    parts.append(
        f"    output[{class_idx}] += tree_{tree_idx}(x);\n"
    )

parts.append("}\n\n")


# Frozen Mahalanobis.
z = np.load(MAHA_PATH)
mean = np.asarray(z["mean"], dtype=np.float64)
inv_cov = np.asarray(z["inv_cov"], dtype=np.float64)
thr = float(z["thr"])

if mean.shape != (6,):
    raise RuntimeError(f"Expected mean shape (6,), got {mean.shape}")

if inv_cov.shape != (6, 6):
    raise RuntimeError(
        f"Expected inverse covariance shape (6,6), got {inv_cov.shape}"
    )


def c_array(name, arr):
    flat = np.asarray(arr).reshape(-1)
    body = ",\n    ".join(fmt(v) for v in flat)
    return (
        f"static const double {name}[{len(flat)}] = {{\n"
        f"    {body}\n"
        f"}};\n"
    )


parts.append(c_array("MAHA_MEAN", mean))
parts.append("\n")
parts.append(c_array("MAHA_INV_COV", inv_cov))
parts.append(
    f"\nstatic const double MAHA_THRESHOLD = {fmt(thr)};\n\n"
)

parts.append(
"""double maha_distance(const double *x)
{
    double diff[CAN_IDS_NUM_FEATURES];
    double dist2 = 0.0;

    for (int i = 0; i < CAN_IDS_NUM_FEATURES; ++i) {
        diff[i] = x[i] - MAHA_MEAN[i];
    }

    for (int i = 0; i < CAN_IDS_NUM_FEATURES; ++i) {
        double s = 0.0;

        for (int j = 0; j < CAN_IDS_NUM_FEATURES; ++j) {
            s += MAHA_INV_COV[i * CAN_IDS_NUM_FEATURES + j] * diff[j];
        }

        dist2 += diff[i] * s;
    }

    /*
     * Numerical round-off can theoretically produce a tiny negative
     * quadratic form near zero. Clamp before sqrt.
     */
    if (dist2 < 0.0 && dist2 > -1e-12) {
        dist2 = 0.0;
    }

    return sqrt(dist2);
}

int maha_predict(const double *x)
{
    return maha_distance(x) > MAHA_THRESHOLD ? 1 : 0;
}

int can_ids_predict(const double *x)
{
    double scores[CAN_IDS_NUM_CLASSES];

    score(x, scores);

    int best = 0;

    for (int i = 1; i < CAN_IDS_NUM_CLASSES; ++i) {
        if (scores[i] > scores[best]) {
            best = i;
        }
    }

    /*
     * Known attack predicted by LightGBM:
     * return immediately.
     */
    if (best != NORMAL_IDX) {
        return CODE_MAP[best];
    }

    /*
     * LightGBM predicted normal:
     * use frozen V2 Mahalanobis detector for unknown anomaly.
     */
    if (maha_predict(x)) {
        return 5;
    }

    return 0;
}
"""
)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("".join(parts))

print("===== FROZEN V2 C EXPORT =====")
print("LightGBM source :", LGB_PATH)
print("Mahalanobis     :", MAHA_PATH)
print("Output          :", OUT_PATH)
print("Features        :", FEATURES)
print("Classes sorted  :", CLASSES_SORTED)
print("NORMAL_IDX      :", NORMAL_IDX)
print("CODE_MAP        :", CODE_MAP)
print("Trees           :", len(trees))
print("Mahalanobis mean:", mean.shape)
print("Mahalanobis inv :", inv_cov.shape)
print("Threshold       :", repr(thr))
print("Output bytes    :", os.path.getsize(OUT_PATH))
print("DONE")
