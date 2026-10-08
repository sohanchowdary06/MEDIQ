# MEDI Q – run locally
    pip install -r requirements.txt
    uvicorn app:app --port 8000        # open http://localhost:8000   (frontend is in static/)

## Make it validated (do this before the demo)
    python calibrate.py ood <folder of normal chest X-rays>     # enables OOD gate -> ood_ref.npz
    python evaluate.py DATA_DIR --fit-temp --stress              # DATA_DIR/pos/*.png, DATA_DIR/neg/*.png
Restart the server afterwards. `evaluate.py` fits temperature on one random half, reports metrics on the other half
(accuracy, precision, recall, specificity, F1, AUC, Brier, ECE before/after calibration, ROC, reliability, abstention rate,
quality-gate catch rate on synthetically degraded X-rays) and writes validation.json (shown in the Validation tab) + results.csv.

## API
GET /health · POST /analyze · GET /history · GET /stats · DELETE /history · GET /validation
The audit log (medi_q.db) stores results and the case ID only – never images or file names.

Educational decision support only – not a medical device. Use public / de-identified images.
