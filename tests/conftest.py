import os

import mlflow
import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_mlflow_tracking(tmp_path_factory):
    """
    Tach MLflow tracking cua test ra mot thu muc tam rieng.

    Ly do: neu khong co MLFLOW_TRACKING_URI, MLflow mac dinh dung ./mlruns lam
    file store. Nhung khi chay `python src/train.py` voi
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db, MLflow van tao ./mlruns de chua
    artifact ma KHONG ghi mlruns/0/meta.yaml (metadata nam trong sqlite).
    Pytest chay sau do se doc ./mlruns nhu file store, thay thieu meta.yaml
    va bao MissingConfigException.

    Fixture nay lam test khong phu thuoc vao bien moi truong hay trang thai
    thu muc ben ngoai - chay giong nhau tren may ca nhan va tren CI runner.
    """
    # Tro vao mot path CHUA ton tai: FileStore chi tu tao experiment mac dinh
    # (experiment ID 0) khi no phai khoi tao thu muc goc tu dau. Neu tro vao mot
    # thu muc rong da ton tai, MLflow bao "Could not find experiment with ID 0".
    tracking_dir = tmp_path_factory.mktemp("mlflow_test") / "mlruns"
    uri = tracking_dir.as_uri()

    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("unit_test")

    yield uri
