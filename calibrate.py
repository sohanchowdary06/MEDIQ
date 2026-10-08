"""python calibrate.py temperature val.npz   # npz with 'logits' and 'labels' (0/1) for the target finding
   python calibrate.py ood folder_of_normal_xrays/"""
import sys, json, glob, numpy as np, cv2, torch
if sys.argv[1] == "temperature":
    d = np.load(sys.argv[2]); z = torch.tensor(d["logits"]).float(); y = torch.tensor(d["labels"]).float()
    T = torch.ones(1, requires_grad=True); opt = torch.optim.LBFGS([T], lr=.1, max_iter=200)
    def step():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(z / T, y); l.backward(); return l
    opt.step(step); json.dump({"temperature": float(T)}, open("calibration.json", "w")); print("T =", float(T))
else:
    from app import embed
    E = []
    for f in glob.glob(sys.argv[2] + "/*"):
        im = cv2.imread(f)
        if im is not None: E.append(embed(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    E = np.stack(E); c = E.mean(0); c /= np.linalg.norm(c)
    th = np.percentile(1 - E @ c, 99)
    np.savez("ood_ref.npz", centroid=c, threshold=th); print("OOD threshold", th, "from", len(E), "images")
