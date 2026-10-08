"""Pure-numpy evaluation maths (no torch needed): calibration, ROC/AUC, reliability, selective prediction."""
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

def nll(z, y, T):
    p = np.clip(sigmoid(z / T), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def fit_temperature(z, y):
    """Temperature scaling: minimise NLL over log T (convex-ish, so ternary search is enough)."""
    lo, hi = -3.0, 3.0
    for _ in range(60):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(z, y, np.exp(a)) < nll(z, y, np.exp(b)): hi = b
        else: lo = a
    return float(np.exp((lo + hi) / 2))

def roc(p, y):
    o = np.argsort(-p); ys = y[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    tpr = np.r_[0, tp / max(tp[-1], 1)]; fpr = np.r_[0, fp / max(fp[-1], 1)]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(101, len(fpr))).astype(int))
    return auc, [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]

def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); ids = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out, ece = [], 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            out.append([round(conf, 3), round(acc, 3), int(m.sum())]); ece += m.mean() * abs(acc - conf)
    return out, float(ece)

def _div(a, b): return float(a / b) if b else 0.0

def summarize(S, y, T, thr=0.7, sd_max=0.12):
    """S: (n, passes) Monte-Carlo-dropout logits for the target finding; y: 0/1 truth."""
    P = sigmoid(S / T); p = P.mean(1); sd = P.std(1, ddof=1)
    pred = (p >= .5).astype(int); y = y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    pre, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    auc, curve = roc(p, y.astype(float)); rel, ece = reliability(p, y)
    _, ece_raw = reliability(sigmoid(S.mean(1)), y)       # T = 1 reference
    conf = np.maximum(p, 1 - p); unc = (conf < thr) | (sd > sd_max); ans = ~unc
    sel = dict(abstain_rate=float(unc.mean()),
               accuracy_answered=_div(int((pred[ans] == y[ans]).sum()), int(ans.sum())),
               recall_answered=_div(int(((pred == 1) & (y == 1) & ans).sum()), int(((y == 1) & ans).sum())),
               accuracy_all=_div(tp + tn, len(y)))
    return dict(n=int(len(y)), n_pos=int(y.sum()), n_neg=int((1 - y).sum()), temperature=round(T, 4), threshold=thr,
                accuracy=_div(tp + tn, len(y)), precision=pre, recall=rec, specificity=_div(tn, tn + fp),
                f1=_div(2 * pre * rec, pre + rec), auc=auc, brier=float(np.mean((p - y) ** 2)),
                ece=ece, ece_uncalibrated=ece_raw, confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn),
                roc=curve, reliability=rel, selective=sel)
