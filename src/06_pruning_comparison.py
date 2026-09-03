import pandas as pd, numpy as np, lightgbm as lgb, time, os
from sklearn.metrics import f1_score

here = os.path.dirname(__file__)
fd = pd.read_parquet(os.path.join(here, "..", "data", "features_v3.parquet"))
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy", "mean_byte", "delta_zscore", "value_zscore", "norm_id"]

train = fd[fd["is_test_region"] == 0]
test = fd[fd["is_test_region"] == 1]
classes = sorted(fd["Label"].unique())
c2i = {c: i for i, c in enumerate(classes)}
Xtr, ytr = train[FEATS], train["Label"].map(c2i)
Xte, yte = test[FEATS], test["Label"].map(c2i)
ds = lgb.Dataset(Xtr, label=ytr)
val = lgb.Dataset(Xte, label=yte, reference=ds)

def run(name, params, rounds):
    m = lgb.train(params, ds, num_boost_round=rounds, valid_sets=[val],
                  callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
    pred = np.argmax(m.predict(Xte), axis=1)
    f1 = f1_score(yte, pred, average="macro")
    path = os.path.join(here, "..", "models", f"{name}.txt")
    m.save_model(path)
    size_kb = os.path.getsize(path) / 1024
    row = Xte.iloc[[0]]
    t0 = time.time()
    for _ in range(300):
        m.predict(row)
    lat = (time.time() - t0) / 300 * 1000
    print(f"[{name}] F1={f1:.4f} size={size_kb:.1f}KB latency={lat:.3f}ms trees={m.num_trees()}")
    return f1, size_kb, lat

print("=== 경량화 전 (Before, 큰 모델) ===")
before = run("before_pruning", {"objective": "multiclass", "num_class": len(classes), "metric": "multi_logloss",
             "num_leaves": 127, "max_depth": -1, "learning_rate": 0.1, "verbose": -1}, 300)

print("=== 경량화 후 (After, 현재 프로덕션 모델) ===")
after = run("after_pruning", {"objective": "multiclass", "num_class": len(classes), "metric": "multi_logloss",
            "num_leaves": 15, "max_depth": 5, "learning_rate": 0.1, "verbose": -1}, 80)

print(f"\n압축률: {before[1]/after[1]:.1f}배 작아짐, 지연시간 {before[2]/after[2]:.1f}배 빨라짐, F1 변화 {after[0]-before[0]:+.4f}")

with open(os.path.join(here, "..", "results", "pruning_comparison.txt"), "w") as f:
    f.write(f"before: F1={before[0]:.4f} size_kb={before[1]:.1f} latency_ms={before[2]:.3f}\n")
    f.write(f"after: F1={after[0]:.4f} size_kb={after[1]:.1f} latency_ms={after[2]:.3f}\n")
    f.write(f"compression_ratio={before[1]/after[1]:.1f}x\n")
