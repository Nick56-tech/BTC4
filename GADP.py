# gmad_bitcoin_csv.py  ─────────────────────────────────────────────────────────
"""
CODE MARCHE
Transformer GMAD (Graph Matching Attention Decoder) pour BTC 5-minutes
Lecture directe du CSV : btc_bybit_jan2021_5m_filled.csv
PyTorch + PyTorch-Lightning ≥ 2.2
"""
import math, torch, torch.nn as nn, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import matplotlib.pyplot as plt

# 1. Activer MPS
# -- 1. Activer MPS si dispo --
if torch.backends.mps.is_available():
    DEVICE = "mps"
    torch.backends.mps.fallback_enabled = False   # ← OK : existe bien
else:
    DEVICE = "cpu"
num_worker=9
# ------------------ 0. HYPERPARAMS ------------------
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 288        # ≈ 24 h de bougies 5 mn
BATCH   = 256
LR      = 3e-4
EPOCHS  = 3
DICT_SZ = 128        # taille du dictionnaire de graph-patterns

# ------------------ 1. DATA -------------------------

FILE = Path("btc_bybit_jan2021_5m_filled.csv")
FILE = Path("btc_bybit_jan2021_1h.csv")

dtype_cols = {
    "open": "float32", "high": "float32", "low": "float32", "close": "float32",
    "volume": "float32", "turnover": "float32", "openInterest": "float32",
    "fundingRate": "float32", "fundingRateTimestamp": "float64"
}
df = pd.read_csv(
    FILE,
    dtype=dtype_cols,
    parse_dates=["timestamp"],
    infer_datetime_format=True
).sort_values("timestamp")

df["ret"] = np.log(df["close"]).diff()
df = df.dropna().reset_index(drop=True)

feat_cols = ["open", "high", "low", "close",
             "volume", "turnover", "openInterest", "fundingRate"]
X = df[feat_cols].pct_change().fillna(0).values.astype(np.float32)
y = df["ret"].shift(-1).fillna(0).values.astype(np.float32)

# normalisation (stats train-set)
split = int(len(df) * 0.7)
mu, sd = X[:split].mean(0), X[:split].std(0) + 1e-8
X = (X - mu) / sd

class WindowedDS(Dataset):
    def __init__(self, X, y, L):
        self.X, self.y, self.L = X, y, L
    def __len__(self):  return len(self.X) - self.L - 1
    def __getitem__(self, i):
        return (self.X[i:i + self.L], self.y[i + self.L])

train_ds = WindowedDS(X[:split], y[:split], SEQ_LEN)
test_ds  = WindowedDS(X[split - SEQ_LEN:], y[split - SEQ_LEN:], SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

# ------------------ 2. MODEL BLOCKS -----------------
class GraphLearner(nn.Module):
    """Apprend un graphe latent par pas de temps"""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x):                     # x: (B,L,D)
        h = x.mean(1)                         # agrégation temporelle
        proj = self.linear(h)                 # (B,D)
        A = torch.sigmoid(proj.unsqueeze(2) - proj.unsqueeze(1))
        return A                              # (B,D,D)

class MatchingPool(nn.Module):
    """Pool de graph-patterns + score d’anomalie (distance Frobenius)"""
    def __init__(self, d, dict_sz=DICT_SZ):
        super().__init__()
        self.register_buffer("P", torch.randn(dict_sz, d, d))
    def forward(self, A):                     # A: (B,D,D)
        diff = ((A.unsqueeze(1) - self.P) ** 2).mean((2, 3))  # (B,dict_sz)
        idx  = diff.argmin(1)
        best = self.P[idx]                    # pattern le + proche
        score = diff.min(1).values            # distance → anomalie
        return best, score

class GMADBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, A):                  # x:(B,L,D), A:(B,D,D)
        gate = A.mean(1)                      # importance par feature
        k = x * gate.unsqueeze(1)             # clés « gated »
        attn_out, _ = self.attn(query=x, key=k, value=x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class GMADTransformer(pl.LightningModule):
    def __init__(self, d_model=64, n_heads=4, n_layers=3, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        self.embed   = nn.Linear(len(feat_cols), d_model)
        self.graph   = GraphLearner(d_model)
        self.match   = MatchingPool(d_model)
        self.blocks  = nn.ModuleList(
            [GMADBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.reg_head = nn.Linear(d_model, 1)
        self.lr = lr
        self.mse = nn.MSELoss()

    # ---- forward : prédiction + score anomalie ----
    def forward(self, x):                     # x:(B,L,D_in)
        z = self.embed(x)                     # (B,L,d_model)
        A = self.graph(z)
        A_ref, score = self.match(A)
        for blk in self.blocks:
            z = blk(z, A_ref)
        out = self.reg_head(z[:, -1])         # dernier token
        return out.squeeze(1), score

    # ---- Lightning boilerplate ----
    def training_step(self, batch, _):
        x, y = batch
        y_hat, score = self(x)
        loss = self.mse(y_hat, y) + 0.1 * score.mean()
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# ------------------ 3. TRAINING ---------------------
model = GMADTransformer()
trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="auto", logger=False)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    precision="16-mixed",
    logger=False
)

trainer.fit(model, train_loader)

# ------------------ 4. INFERENCE & BACKTEST ---------
@torch.no_grad()
def predict(loader):
    preds, scores = [], []
    for x, _ in loader:
        p, s = model(x.to(model.device))
        preds.append(p.cpu())
        scores.append(s.cpu())
    return torch.cat(preds), torch.cat(scores)

preds, scores = predict(test_loader)
test_idx = df.index[split + 1:]               # +1 à cause du diff
test_df = df.loc[test_idx].copy()
test_df["pred_ret"] = preds.numpy()
test_df["anom"]     = scores.numpy()

# règle de trading : long si pred_ret>0 & anomalie<70ᵉ pct
thr = np.percentile(test_df["anom"], 70)
signal = ((test_df["pred_ret"] > 0) & (test_df["anom"] < thr)).astype(int)
test_df["strat_ret"] = signal * test_df["ret"]
test_df["eq_curve"]  = test_df["strat_ret"].cumsum().apply(np.exp)
test_df["bh_curve"]  = test_df["ret"].cumsum().apply(np.exp)

def sharpe(x, freq=365 * 24 * 12):
    return x.mean() * freq / (x.std() * math.sqrt(freq) + 1e-8)
def maxdd(curve):
    roll = curve.cummax()
    return ((curve / roll) - 1).min()

print(f"Sharpe  : {sharpe(test_df['strat_ret']):.2f}")
print(f"Max DD : {maxdd(test_df['eq_curve'])*100:.1f} %")

plt.figure(figsize=(9, 4))
plt.plot(test_df["eq_curve"], label="GMAD strategy")
plt.plot(test_df["bh_curve"], "--", label="Buy & Hold")
plt.title("Backtest BTCUSDT 5-mn")
plt.ylabel("Equity (index=1)")
plt.legend()
plt.tight_layout()
plt.show()

out = pd.DataFrame({
    "timestamp": df.loc[test_idx, "timestamp"].values,
    "pred_ret_log": preds.numpy(),                       # r̂(log)
    "pred_ret_pct": (np.exp(preds.numpy()) - 1) * 100,   # r̂ en %
    "real_ret_log": df.loc[test_idx, "ret"].values,
    "real_ret_pct": (np.exp(df.loc[test_idx, "ret"].values) - 1) * 100,
    "anomaly": scores.numpy()
})

out.to_csv("gmad_predictions_detail.csv", index=False)
print("→ CSV sauvegardé : gmad_predictions_detail.csv")
# -----------------------------------------------------------------------------
git status