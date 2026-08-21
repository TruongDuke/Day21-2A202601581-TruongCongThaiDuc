import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

EVAL_THRESHOLD = 0.70

# Bonus 5: neu mot lop chiem it hon ty le nay thi in canh bao lech lac du lieu.
MIN_CLASS_RATIO = 0.10

# Bonus 2: cac thuat toan ho tro qua tham so `model_type` trong params.yaml.
MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}

CLASS_NAMES = {0: "thap", 1: "trung_binh", 2: "cao"}


def build_model(params: dict):
    """
    Bonus 2: chon thuat toan theo params["model_type"], mac dinh random_forest.

    Moi thuat toan co tap sieu tham so khac nhau (vi du logistic_regression
    khong nhan n_estimators), nen chi truyen nhung tham so ma thuat toan do
    thuc su chap nhan va canh bao ro nhung tham so bi bo qua.

    Tra ve: (model_type, estimator)
    """
    p = dict(params)
    model_type = p.pop("model_type", "random_forest")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"model_type '{model_type}' khong ho tro. "
            f"Chon mot trong: {sorted(MODEL_REGISTRY)}"
        )

    cls = MODEL_REGISTRY[model_type]
    accepted = cls().get_params().keys()
    kwargs = {k: v for k, v in p.items() if k in accepted}

    ignored = sorted(set(p) - set(kwargs))
    if ignored:
        print(f"CANH BAO: {model_type} khong nhan cac tham so {ignored} - da bo qua.")

    estimator = cls(**kwargs, random_state=42)

    # LogisticRegression can chuan hoa dac trung, neu khong se hoi tu rat kem
    # tren tap Wine Quality (cac cot lech thang do rat lon).
    if model_type == "logistic_regression":
        estimator.set_params(max_iter=kwargs.get("max_iter", 1000))
        estimator = make_pipeline(StandardScaler(), estimator)

    return model_type, estimator


def check_label_distribution(y_train) -> dict:
    """
    Bonus 5: tinh ty le phan phoi nhan va canh bao neu co lop qua it mau.

    Tra ve: dict {ten_lop: ty_le} de ghi vao outputs/metrics.json.
    """
    ratios = y_train.value_counts(normalize=True).sort_index()
    distribution = {}

    print("Phan phoi nhan trong tap huan luyen:")
    for label, ratio in ratios.items():
        name = CLASS_NAMES.get(int(label), str(label))
        distribution[f"class_{int(label)}_{name}"] = round(float(ratio), 4)
        flag = "  <-- CANH BAO: duoi 10%" if ratio < MIN_CLASS_RATIO else ""
        print(f"  lop {int(label)} ({name:<11}): {ratio:.2%}{flag}")

    thin = [lb for lb, r in ratios.items() if r < MIN_CLASS_RATIO]
    if thin:
        print(
            f"CANH BAO LECH LAC DU LIEU: cac lop {thin} chiem duoi "
            f"{MIN_CLASS_RATIO:.0%} tong mau. Mo hinh se du doan kem tren cac lop nay."
        )
    else:
        print(f"Khong co lop nao duoi {MIN_CLASS_RATIO:.0%} - phan phoi nhan on dinh.")

    return distribution


def write_report(y_eval, preds, model_type: str, acc: float, f1: float) -> None:
    """
    Bonus 3: ghi confusion matrix va precision/recall tung lop ra outputs/report.txt.
    """
    labels = sorted(y_eval.unique())
    cm = confusion_matrix(y_eval, preds, labels=labels)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_eval, preds, labels=labels, zero_division=0
    )

    lines = [
        "BAO CAO HIEU SUAT MO HINH",
        "=" * 52,
        f"Thuat toan : {model_type}",
        f"Accuracy   : {acc:.4f}",
        f"F1 (weighted): {f1:.4f}",
        "",
        "CONFUSION MATRIX (hang = nhan thuc, cot = nhan du doan)",
        "-" * 52,
    ]

    header = "          " + "".join(f"{CLASS_NAMES.get(int(l), l):>13}" for l in labels)
    lines.append(header)
    for i, label in enumerate(labels):
        name = CLASS_NAMES.get(int(label), str(label))
        row = "".join(f"{int(v):>13}" for v in cm[i])
        lines.append(f"{name:>10}{row}")

    lines += [
        "",
        "PRECISION / RECALL THEO TUNG LOP",
        "-" * 52,
        f"{'lop':<14}{'precision':>11}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for i, label in enumerate(labels):
        name = f"{int(label)} ({CLASS_NAMES.get(int(label), label)})"
        lines.append(
            f"{name:<14}{precision[i]:>11.4f}{recall[i]:>10.4f}"
            f"{f1_per_class[i]:>10.4f}{int(support[i]):>10}"
        )

    os.makedirs("outputs", exist_ok=True)
    report = "\n".join(lines) + "\n"
    with open("outputs/report.txt", "w") as f:
        f.write(report)
    print("\n" + report)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho mo hinh.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        # Bonus 5: kiem tra phan phoi nhan TRUOC khi huan luyen
        distribution = check_label_distribution(y_train)

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)
        mlflow.log_param("n_train_samples", len(df_train))

        # TODO 4: Khoi tao va huan luyen mo hinh (Bonus 2: chon theo model_type)
        model_type, model = build_model(params)
        print(f"Huan luyen {model_type} tren {len(df_train)} mau...")
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for name, ratio in distribution.items():
            mlflow.log_metric(f"dist_{name}", ratio)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Bonus 3: bao cao confusion matrix + precision/recall tung lop
        write_report(y_eval, preds, model_type, acc, f1)
        mlflow.log_artifact("outputs/report.txt")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "model_type": model_type,
                    "n_train_samples": len(df_train),
                    "label_distribution": distribution,
                },
                f,
                indent=2,
            )

        # TODO 9: Luu mo hinh ra file models/model.pkl
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
