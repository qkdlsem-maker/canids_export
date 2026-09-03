"""
item5 해결: 3번(솔루션 유형) 요구사항 - 1D-CNN, LightGBM, TinyTransformer 성능 비교
GPU 서버(gpu0/gpu1)에서 실행. 실행 전:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    (또는 서버에 이미 깔린 CUDA 버전에 맞는 torch)

입력: 시계열 원시 시퀀스 (CAN ID + DATA[0..7], 총 9채널) 슬라이딩 윈도우
      -> 탭형 피처(freq/entropy 등) 대신 raw 시퀀스를 직접 넣어서
         1D-CNN/Transformer가 스스로 패턴을 학습하게 함 (LightGBM과 공정 비교를 위해
         "같은 원시 정보량"에서 출발하되, 모델 구조 자체의 표현력을 비교하는 게 목적)

출력: results/gpu_comparison.txt (F1, 지연시간, 모델크기, 파라미터수)
"""
import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW = 20
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Car_Hacking_5pct.csv")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "gpu_comparison.txt")

# ---------- 1. 데이터 로드 및 원시 시퀀스 구성 (leakage 방지: Label별 시간순 split) ----------
def load_sequences():
    df = pd.read_csv(DATA_PATH)
    classes = sorted(df["Label"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    ids = df["CAN ID"].values.astype(np.float32) / 2048.0
    payload = df[[f"DATA[{i}]" for i in range(8)]].values.astype(np.float32) / 255.0
    x_all = np.concatenate([ids[:, None], payload], axis=1)  # (N, 9)
    y_all = df["Label"].map(c2i).values

    n = len(df)
    is_test = np.zeros(n, dtype=bool)
    for lbl in df["Label"].unique():
        idxs = np.where(df["Label"].values == lbl)[0]
        cut = int(len(idxs) * 0.7)
        is_test[idxs[cut:]] = True
    return x_all, y_all, is_test, classes


class CANWindowDataset(Dataset):
    """window: 직전 WINDOW개 메시지의 raw feature 시퀀스 -> 현재 메시지 라벨"""
    def __init__(self, x_all, y_all, indices, window=WINDOW):
        self.x_all, self.y_all, self.indices, self.window = x_all, y_all, indices, window

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        start = max(0, i - self.window + 1)
        seq = self.x_all[start:i + 1]
        if len(seq) < self.window:
            pad = np.zeros((self.window - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        return torch.tensor(seq, dtype=torch.float32), int(self.y_all[i])


# ---------- 2. 모델 정의 ----------
class TinyCNN(nn.Module):
    """경량 1D-CNN: Conv1d 2단 + GAP + Linear"""
    def __init__(self, in_ch=9, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):  # x: (B, W, C) -> (B, C, W)
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return self.fc(h)


class TinyTransformer(nn.Module):
    """경량 Transformer: 1 encoder layer, d_model=32, nhead=4"""
    def __init__(self, in_ch=9, num_classes=5, d_model=32, window=WINDOW):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=64,
                                           batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.proj(x) + self.pos
        h = self.encoder(h)
        h = h[:, -1, :]  # 마지막(현재) 타임스텝
        return self.fc(h)


# ---------- 3. 학습/평가 루프 ----------
def train_and_eval(model, name, train_loader, test_loader, classes, epochs=3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        print(f"[{name}] epoch {ep+1}/{epochs} loss={loss.item():.4f}")
    train_time = time.time() - t0

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(DEVICE))
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    f1 = f1_score(trues, preds, average="macro")

    # CPU 기준 단건 추론 지연시간 (실차 ECU는 GPU 없음 -> CPU로 측정)
    model_cpu = model.to("cpu").eval()
    single = next(iter(test_loader))[0][:1]
    n_reps = 200
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_reps):
            model_cpu(single)
    latency_ms = (time.time() - t0) / n_reps * 1000

    ckpt_path = f"/tmp/{name}.pt"
    torch.save(model_cpu.state_dict(), ckpt_path)
    size_kb = os.path.getsize(ckpt_path) / 1024
    n_params = sum(p.numel() for p in model.parameters())

    print(f"[{name}] F1={f1:.4f} latency={latency_ms:.3f}ms size={size_kb:.1f}KB params={n_params:,} train_time={train_time:.1f}s")
    return {"name": name, "f1": f1, "latency_ms": latency_ms, "size_kb": size_kb, "params": n_params}


def main():
    x_all, y_all, is_test, classes = load_sequences()
    idx_train = np.where(~is_test)[0]
    idx_test = np.where(is_test)[0]
    # 서브샘플링 (전체 80만건 시퀀스 생성은 무거움 - 필요시 조정)
    rng = np.random.default_rng(42)
    idx_train = rng.choice(idx_train, size=min(150000, len(idx_train)), replace=False)
    idx_test = rng.choice(idx_test, size=min(50000, len(idx_test)), replace=False)

    train_ds = CANWindowDataset(x_all, y_all, idx_train)
    test_ds = CANWindowDataset(x_all, y_all, idx_test)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    results = []
    results.append(train_and_eval(TinyCNN(num_classes=len(classes)), "1D-CNN", train_loader, test_loader, classes, epochs=30))
    results.append(train_and_eval(TinyTransformer(num_classes=len(classes)), "TinyTransformer", train_loader, test_loader, classes, epochs=30))

    print("\n=== 최종 비교 (LightGBM 수치는 results/final_summary.txt 참고: F1=1.0000, size=709KB, latency=0.28ms) ===")
    for r in results:
        print(r)

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
