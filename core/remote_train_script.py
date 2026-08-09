# Remote GPU training script - reads data from stdin
import sys, os, time, subprocess, json
print("START|script loaded", flush=True)

# SSH 环境可能找不到用户安装的包, 手动加路径
_user_site = os.path.expanduser("~/.local/lib")
if os.path.isdir(_user_site):
    for _ver in os.listdir(_user_site):
        _sp = os.path.join(_user_site, _ver, "site-packages")
        if os.path.isdir(_sp) and _sp not in sys.path:
            sys.path.insert(0, _sp)

# Auto-install BEFORE importing numpy/torch (they may be missing)
for pkg, imp in [("numpy", "numpy"), ("scikit-learn", "sklearn"), ("torch", "torch")]:
    try:
        __import__(imp)
    except ImportError:
        print(f"INSTALL|{pkg}...", flush=True)
        for pip in ["python3 -m pip", "pip3", "pip"]:
            try:
                r = subprocess.run(pip.split() + ["install", pkg, "-q", "--user"],
                                   capture_output=True, text=True, timeout=180)
                if r.returncode == 0:
                    __import__(imp)
                    break
            except: continue
        else:
            print(f"FATAL|cannot install {pkg}", flush=True)
            sys.exit(1)

import numpy as np

# ==== Params from args ====
params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
EPOCHS = params.get("epochs", 30)
BATCH_SIZE = params.get("batch_size", 512)
LR = params.get("lr", 0.001)
HIDDEN_DIM = params.get("hidden_dim", 128)
NUM_LAYERS = params.get("num_layers", 2)
NUM_HEADS = params.get("num_heads", 4)
GPU_ID = params.get("gpu_id", 0)
OUTPUT = params.get("output", "/tmp/best_model.pt")

# ==== Read data from stdin ====
data = []
n_read = 0
for line in sys.stdin:
    line = line.strip()
    if line and not line.startswith("#"):
        parts = line.split(",")
        if len(parts) >= 2:
            data.append([float(p) for p in parts])
        n_read += 1
        if n_read % 50000 == 0:
            print(f"READ|{n_read} lines...", flush=True)
data = np.array(data)
print(f"DATA|{len(data)} rows, {data.shape[1]} cols", flush=True)

# Auto-detect mode: 4 cols = TXT raw, >4 cols = pre-extracted features
if data.shape[1] <= 4:
    # TXT mode: raw data, need feature extraction
    lon, lat, intensity, seq = data[:,0], data[:,1], data[:,2], data[:,3]
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    features = np.column_stack([
        intensity, seq,
        np.abs(np.diff(intensity, prepend=intensity[0])),
        np.convolve(intensity, np.ones(5)/5, mode="same"),
        lon, lat,
    ])
    scaler = StandardScaler()
    feat = scaler.fit_transform(features)
    labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(feat)
    print(f"FEAT|TXT→{feat.shape[1]} features")
else:
    # SEG mode: pre-extracted features (last col = labels)
    feat = data[:, :-1]
    labels = data[:, -1].astype(int)
    print(f"FEAT|SEG pre-extracted: {feat.shape[1]} features")

# ==== MSC-Transformer ====
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=256):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))
    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]

class MSCTransformer(nn.Module):
    # 与本地 models/msc_transformer.py 完全一致
    def __init__(self, input_dim, num_classes=5, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1, dim_feedforward=None):
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        super().__init__()
        self.input_embedding = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.pos_encoding = PositionalEncoding(hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.attention_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//4), nn.Tanh(), nn.Linear(hidden_dim//4, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, hidden_dim//4), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(hidden_dim//4, num_classes))
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        attn = torch.softmax(self.attention_pool(x), dim=1)
        x = (x * attn).sum(dim=1)
        return self.classifier(x)

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
print(f"DEVICE|{device}", flush=True)
model = MSCTransformer(input_dim=feat.shape[1], hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS).to(device)

# ==== Training ====
n = len(feat)
split = int(n * 0.8)
idx = np.random.permutation(n)
X_tr = torch.FloatTensor(feat[idx[:split]]).to(device)
y_tr = torch.LongTensor(labels[idx[:split]]).to(device)
X_vl = torch.FloatTensor(feat[idx[split:]]).to(device)
y_vl = torch.LongTensor(labels[idx[split:]]).to(device)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
opt = optim.AdamW(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()
best_loss = float("inf")
t0 = time.time()

for epoch in range(EPOCHS):
    model.train(); tr_loss = 0; tr_ok = 0; tr_n = 0
    for bx, by in loader:
        opt.zero_grad()
        loss = crit(model(bx), by)
        loss.backward()
        opt.step()
        tr_loss += loss.item() * len(bx)
        tr_ok += (model(bx).argmax(1) == by).sum().item()
        tr_n += len(bx)

    model.eval()
    with torch.no_grad():
        vl_out = model(X_vl)
        vl_loss = crit(vl_out, y_vl).item()
        vl_acc = (vl_out.argmax(1) == y_vl).float().mean().item()
    tr_acc = tr_ok / tr_n; tr_loss /= tr_n
    if vl_loss < best_loss:
        best_loss = vl_loss
        torch.save({
            "model_state": model.state_dict(),
            "input_dim": feat.shape[1], "num_classes": 5,
            "hidden_dim": HIDDEN_DIM, "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS, "dropout": 0.1,
            "is_trained": True,
        }, OUTPUT)

    eta = (time.time() - t0) / (epoch + 1) * (EPOCHS - epoch - 1) if epoch > 0 else 0
    print(f"PROGRESS|{epoch+1}|{tr_loss:.4f}|{vl_loss:.4f}|{tr_acc:.4f}|{vl_acc:.4f}|{eta:.0f}")

print(f"DONE|{best_loss:.4f}")
