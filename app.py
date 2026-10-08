"""MEDI Q backend v2: DenseNet-121 + Grad-CAM + OpenCV quality gate + EfficientNet OOD
+ temperature scaling + MC dropout + rule-based clinical safety layer + SQLite audit log.
Educational use only - not a medical device."""
import base64, json, os, sqlite3, threading, time, uuid
import cv2, numpy as np, torch, torch.nn.functional as F, torchvision
import torchxrayvision as xrv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BYTES = 15 * 1024 * 1024
LOCK = threading.Lock()                       # one inference at a time (shared model + autograd state)

# 1) X-ray classifier
model = xrv.models.DenseNet(weights="densenet121-res224-all").to(dev).eval()
TARGETS = [model.pathologies.index(p) for p in ("Pneumonia", "Lung Opacity", "Consolidation") if p in model.pathologies]

# 2) Temperature scaling (written by evaluate.py --fit-temp or calibrate.py; 1.0 = uncalibrated)
cal_path = os.path.join(HERE, "calibration.json")
TEMP = json.load(open(cal_path))["temperature"] if os.path.exists(cal_path) else 1.0

# 3) OOD: EfficientNet-B0 embeddings, cosine distance to in-distribution centroid
eff = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
eff.classifier = torch.nn.Identity(); eff = eff.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
ref_path = os.path.join(HERE, "ood_ref.npz")
OOD = np.load(ref_path) if os.path.exists(ref_path) else None

@torch.no_grad()
def embed(rgb):
    x = torch.from_numpy(cv2.resize(rgb, (224, 224))).float().div(255).permute(2, 0, 1)[None].to(dev)
    with LOCK: e = eff((x - MEAN) / STD)[0].cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-8)

# 4) Image quality
def quality(rgb, gray):
    h, w = gray.shape
    g = cv2.resize(gray, (512, 512))
    q = dict(width=w, height=h, blur=float(cv2.Laplacian(g, cv2.CV_64F).var()), brightness=float(gray.mean()),
             contrast=float(gray.std()), saturation=float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].mean()))
    r = []
    if min(h, w) < 224: r.append("Low resolution")
    if q["blur"] < 25: r.append("Blurry")
    if q["brightness"] < 40: r.append("Too dark")
    if q["brightness"] > 215: r.append("Overexposed")
    if q["contrast"] < 20: r.append("Low contrast")
    if q["saturation"] > 35: r.append("Colour image - not a radiograph")
    return q, r

# 5) Inference + Grad-CAM
def prep(gray):
    return torch.from_numpy(xrv.datasets.normalize(cv2.resize(gray, (224, 224)).astype(np.float32), 255))[None, None].to(dev)

def feats_pooled(x):
    A = F.relu(model.features(x))
    return A, F.adaptive_avg_pool2d(A, (1, 1)).flatten(1)

@torch.no_grad()
def infer(gray, target="auto", mc_n=20):
    """Returns (x, k, logit, mc_logits[mc_n]) for the chosen finding. Probabilities are derived by the caller
    so any temperature can be applied (used by evaluate.py)."""
    x = prep(gray)
    with LOCK:
        _, pooled = feats_pooled(x)
        logits = model.classifier(pooled)[0]
        k = model.pathologies.index(target) if target in model.pathologies else max(TARGETS, key=lambda i: logits[i].item())
        mcl = torch.stack([model.classifier(F.dropout(pooled, .3, True))[0, k] for _ in range(max(2, min(mc_n, 100)))])
    return x, k, float(logits[k]), mcl.cpu().numpy()

def gradcam(x, k, base):
    with LOCK, torch.enable_grad():
        A, pooled = feats_pooled(x)
        g = torch.autograd.grad(model.classifier(pooled)[0, k], A)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * A).sum(1))[0].detach().cpu().numpy()
    cam = cv2.resize(cam, (256, 256)); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), .55, heat, .45, 0)
    y, xx = np.unravel_index(cam.argmax(), cam.shape)
    where = f"{'right' if xx < 128 else 'left'} {'upper' if y < 85 else 'mid' if y < 170 else 'lower'} lung field (patient orientation)"
    return base64.b64encode(cv2.imencode(".png", over)[1]).decode(), where

# 6) Clinical context + rule-based safety layer
def clinical(spo2, temp, hr, age, sy):
    s = 0
    if age >= 65: s += 1
    if temp >= 38: s += 1
    if 0 < spo2 < 94: s += 3 if spo2 < 90 else 2
    if hr > 100: s += 1
    if len(sy) >= 3: s += 1
    red = (0 < spo2 < 92) or ("Chest pain" in sy and 0 < spo2 < 95)
    return dict(score=s, level="High" if s >= 4 else "Moderate" if s >= 2 else "Low", red_flag=red)

def triage(status, p, reasons, clin):
    extra = (" - image issues: " + ", ".join(reasons)) if reasons else ""
    if clin["red_flag"]: return "Escalate to Expert", "Clinical red flag (low SpO2 / symptom combination) overrides image result" + extra
    if clin["level"] == "High": return "Escalate to Expert", f"High clinical risk score ({clin['score']}) overrides image result" + extra
    if status == "poor": return "Retake Image", ", ".join(reasons)
    if status == "uncertain": return "Escalate to Expert", "Low calibrated confidence or high Monte-Carlo-dropout variance"
    if status == "positive":
        return ("High Risk – prioritise specialist review", f"Calibrated probability {p:.0%}") if p >= .85 else ("Moderate Risk – clinician review", f"Calibrated probability {p:.0%}")
    return "Low Risk – routine follow-up", "No opacity pattern above threshold"

# 7) Audit log (SQLite). Stores results + the case ID only - never images or file names.
DB = os.path.join(HERE, "medi_q.db")
def sql(q, args=(), many=False):
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    try:
        cur = c.execute(q, args); rows = cur.fetchall() if many else None; c.commit(); return rows
    finally: c.close()
sql("""CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY, ts REAL, case_id TEXT, status TEXT, finding TEXT, label TEXT,
       prob REAL, conf REAL, mc_std REAL, triage TEXT, reason TEXT, escalated INTEGER, ood_dist REAL)""")

app = FastAPI(title="MEDI Q")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return dict(ok=True, model="DenseNet-121 (TorchXRayVision)", device=dev, temperature=TEMP, ood_calibrated=OOD is not None,
                validated=os.path.exists(os.path.join(HERE, "validation.json")))

@app.get("/history")
def history(limit: int = 50):
    return [dict(r) for r in sql("SELECT * FROM analyses ORDER BY ts DESC LIMIT ?", (max(1, min(limit, 500)),), many=True)]

@app.get("/stats")
def stats():
    r = {s: n for s, n in sql("SELECT status, COUNT(*) FROM analyses GROUP BY status", many=True)}
    esc = sql("SELECT COUNT(*) FROM analyses WHERE escalated=1", many=True)[0][0]
    return dict(pos=r.get("positive", 0), neg=r.get("negative", 0), unc=r.get("uncertain", 0), poor=r.get("poor", 0), esc=esc)

@app.delete("/history")
def clear_history():
    sql("DELETE FROM analyses"); return dict(ok=True)

@app.get("/validation")
def validation():
    p = os.path.join(HERE, "validation.json")
    if not os.path.exists(p): raise HTTPException(404, "No validation yet - run: python evaluate.py <data_dir>")
    return json.load(open(p))

@app.post("/analyze")
def analyze(file: UploadFile = File(...), spo2: float = Form(0), sy: str = Form(""), target: str = Form("auto"), mc: int = Form(20),
            heatmap: bool = Form(True), ood: bool = Form(True), qc: bool = Form(True), thr: float = Form(.7),
            case_id: str = Form(""), age: float = Form(0), temp: float = Form(0), hr: float = Form(0)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES: raise HTTPException(413, "Image larger than 15 MB")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None: raise HTTPException(400, "Could not decode image (use PNG/JPEG)")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    symptoms = [s.strip() for s in sy.split(",") if s.strip()]
    clin = clinical(spo2, temp, hr, age, symptoms)
    q, reasons = quality(rgb, gray)
    if not qc: reasons = []
    ood_dist = None
    if OOD is not None and ood:
        ood_dist = float(1 - embed(rgb) @ OOD["centroid"])
        if ood_dist > float(OOD["threshold"]): reasons.append("Out-of-distribution (not like training X-rays)")
    out = dict(id=uuid.uuid4().hex[:12], case_id=case_id[:32], quality=q, temperature=TEMP, clinical=clin,
               ood_dist=round(ood_dist, 3) if ood_dist is not None else "n/a")
    if reasons:
        t, why = triage("poor", 0, reasons, clin)
        out.update(status="poor", reasons=reasons, confidence=0, probability=0, finding="Not assessable", label="-", evidence="",
                   heatmap=None, mc_std=0, triage=t, reason=why)
    else:
        x, k, _, mcl = infer(gray, target, mc)
        P = 1 / (1 + np.exp(-mcl / TEMP))                          # temperature scaling on every MC pass
        p, std = float(P.mean()), float(P.std(ddof=1)); conf = max(p, 1 - p)
        status = "uncertain" if (conf < thr or std > .12) else "positive" if p >= .5 else "negative"
        hm, where = gradcam(x, k, cv2.resize(gray, (256, 256))) if heatmap else (None, "")
        t, why = triage(status, p, [], clin)
        ev = "Heatmap disabled in analysis options." if not heatmap else f"Grad-CAM peak in the {where}." if status != "negative" \
            else f"Highest attention in the {where}; no finding above threshold."
        out.update(status=status, reasons=[], label=model.pathologies[k], probability=round(p, 3), confidence=round(conf, 3),
                   mc_std=round(std, 3), heatmap=hm, triage=t, reason=why, evidence=ev,
                   finding={"positive": "Opacity pattern detected", "negative": "No opacity pattern detected", "uncertain": "Possible opacity (uncertain)"}[status])
    sql("INSERT INTO analyses VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (out["id"], time.time(), out["case_id"], out["status"], out["finding"], out["label"], out["probability"], out["confidence"],
         out["mc_std"], out["triage"], out["reason"], int(out["triage"].startswith("Escalate")), ood_dist))
    return out

if os.path.isdir(os.path.join(HERE, "static")):
    app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
