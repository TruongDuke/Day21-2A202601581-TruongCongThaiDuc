from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import shutil

app = FastAPI()

# Ba che do tai model, uu tien tu tren xuong:
#   1. DagsHub / S3-compatible : can DAGSHUB_USER, DAGSHUB_REPO, DAGSHUB_TOKEN
#   2. GCS                     : can GCS_BUCKET + GOOGLE_APPLICATION_CREDENTIALS
#   3. Local                   : dung truc tiep models/model.pkl do src/train.py sinh ra
# Cac bien nay duoc dat trong systemd service tren VM (Buoc 2). Che do local
# giup test /health va /predict tren may ca nhan khi chua co credentials cloud.
DAGSHUB_USER = os.environ.get("DAGSHUB_USER")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN")
GCS_BUCKET = os.environ.get("GCS_BUCKET")

MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")
LOCAL_MODEL_PATH = "models/model.pkl"

LABELS = {0: "thap", 1: "trung_binh", 2: "cao"}
N_FEATURES = 12


def download_model():
    """
    Tai file model.pkl tu cloud storage ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # --- Che do 1: DagsHub Storage (S3-compatible) ---
    if DAGSHUB_USER and DAGSHUB_REPO and DAGSHUB_TOKEN:
        # TODO 1: Tao client
        import boto3
        client = boto3.client(
            "s3",
            endpoint_url=f"https://dagshub.com/api/v1/repo-buckets/s3/{DAGSHUB_USER}",
            aws_access_key_id=DAGSHUB_TOKEN,
            aws_secret_access_key=DAGSHUB_TOKEN,
            region_name="us-east-1",
        )

        # TODO 2 + 3: Lay bucket/key tuong ung va tai file model xuong may
        client.download_file(Bucket=DAGSHUB_REPO, Key=MODEL_KEY, Filename=MODEL_PATH)

        # TODO 4: In thong bao thanh cong
        print(f"Model da duoc tai xuong tu DagsHub ({DAGSHUB_REPO}/{MODEL_KEY}).")
        return

    # --- Che do 2: Google Cloud Storage ---
    if GCS_BUCKET:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print("Model da duoc tai xuong tu GCS.")
        return

    # --- Che do 3: local ---
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise RuntimeError(
            f"Khong co credentials cloud va cung khong tim thay {LOCAL_MODEL_PATH}. "
            "Chay `python src/train.py` truoc, hoac dat cac bien DAGSHUB_* / GCS_BUCKET."
        )
    shutil.copyfile(LOCAL_MODEL_PATH, MODEL_PATH)
    print(f"Che do local: dung {LOCAL_MODEL_PATH}.")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    # TODO 5: Tra ve dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    # TODO 6: Kiem tra so luong dac trung.
    if len(req.features) != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Can dung {N_FEATURES} dac trung, nhan duoc {len(req.features)}.",
        )

    # TODO 7: Goi model.predict([req.features]) de lay ket qua du doan.
    pred = int(model.predict([req.features])[0])

    # TODO 8: Tra ve dict chua "prediction" (int) va "label" (string).
    return {"prediction": pred, "label": LABELS[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
