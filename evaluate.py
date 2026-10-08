"""Validate MEDI Q on a labelled, de-identified test set and write what the dashboard shows.

  python evaluate.py DATA_DIR [--target Pneumonia|auto] [--fit-temp] [--thr 0.7] [--mc 20] [--stress]

DATA_DIR layout:   DATA_DIR/pos/*.png|jpg   (finding present)     DATA_DIR/neg/*.png|jpg   (normal)

--fit-temp  fits the temperature on a random half and reports metrics on the OTHER half (no leakage),
            then saves calibration.json (restart the server to use it).
Outputs: validation.json (served at /validation), results.csv (truth,pred,prob,std,file)."""
import argparse, csv, glob, json, os, time
import cv2, numpy as np
import metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("--target", default="auto"); ap.add_argument("--thr", type=float, default=.7)
ap.add_argument("--mc", type=int, default=20); ap.add_argument("--fit-temp", action="store_true"); ap.add_argument("--stress", action="store_true")
a = ap.parse_args()
import app                                                     # loads the model once

files, y = [], []
for lab, d in ((1, "pos"), (0, "neg")):
    for f in sorted(glob.glob(os.path.join(a.data, d, "*"))):
        if cv2.imread(f, cv2.IMREAD_GRAYSCALE) is not None: files.append(f); y.append(lab)
y = np.array(y); assert len(y) > 10 and y.min() == 0 and y.max() == 1, "need images in both pos/ and neg/"
Z, S, grays = [], [], []
for i, f in enumerate(files):
    g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    _, k, z, mcl = app.infer(g, a.target, a.mc); Z.append(z); S.append(mcl)
    if a.stress and len(grays) < 200: grays.append(g)
    if i % 25 == 0: print(f"{i}/{len(files)}", end="\r")
Z, S = np.array(Z), np.stack(S)

rng = np.random.RandomState(0); idx = rng.permutation(len(y))
if a.fit_temp:
    fit, ev = idx[: len(y) // 2], idx[len(y) // 2:]
    T = M.fit_temperature(Z[fit], y[fit].astype(float)); json.dump({"temperature": T}, open(os.path.join(app.HERE, "calibration.json"), "w"))
    split = f"temperature fitted on {len(fit)} images, metrics on {len(ev)} held-out images"
else:
    ev, T, split = idx, app.TEMP, "all images (temperature not fitted on this run)"
res = M.summarize(S[ev], y[ev], T, a.thr); res.update(target=a.target, split=split, generated=time.strftime("%Y-%m-%d %H:%M"))

if a.stress:   # do the quality gates catch synthetic degradations of real X-rays?
    deg = {"blur": lambda g: cv2.GaussianBlur(g, (0, 0), 9), "too_dark": lambda g: (g * .2).astype(np.uint8),
           "overexposed": lambda g: np.clip(g.astype(float) * .3 + 190, 0, 255).astype(np.uint8), "colour": lambda g: cv2.applyColorMap(g, cv2.COLORMAP_JET)}
    st = {}
    for n, fn in deg.items():
        caught = 0
        for g in grays:
            d = fn(g); rgb = d if d.ndim == 3 else cv2.cvtColor(d, cv2.COLOR_GRAY2RGB)
            caught += bool(app.quality(rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))[1])
        st[n] = round(caught / max(len(grays), 1), 3)
    res["stress_gate_catch_rate"] = st; res["stress_n"] = len(grays)

json.dump(res, open(os.path.join(app.HERE, "validation.json"), "w"), indent=1)
P = M.sigmoid(S / T); p = P.mean(1)
with open(os.path.join(app.HERE, "results.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["truth", "pred", "prob", "std", "file"])
    for i in ev: w.writerow([y[i], int(p[i] >= .5), round(p[i], 4), round(P[i].std(ddof=1), 4), os.path.basename(files[i])])
print(split); print({k: round(v, 3) for k, v in res.items() if isinstance(v, float)})
