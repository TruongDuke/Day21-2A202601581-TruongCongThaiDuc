# Báo Cáo Lab MLOps — Day 21: CI/CD cho AI Systems

Họ tên: Trương Công Thái Đức · Khóa K3 · AIInAction, VinUni

---

## 1. Bộ siêu tham số đã chọn và lý do

```yaml
n_estimators: 500
max_depth: null          # không giới hạn độ sâu
min_samples_split: 5
model_type: random_forest
```

**Kết quả 20 thí nghiệm trên MLflow** (tập `train_phase1.csv`, 2998 mẫu; đánh giá trên `eval.csv`, 500 mẫu):

| n_estimators | max_depth | min_samples_split | accuracy | f1_score |
|---|---|---|---|---|
| 1000 | None | 2 | 0.6840 | 0.6826 |
| **500** | **None** | **5** | **0.6800** | **0.6784** |
| 600 | 30 | 2 | 0.6760 | 0.6749 |
| 300 | 20 | 2 | 0.6740 | 0.6729 |
| 200 | None | 2 | 0.6720 | 0.6711 |
| 200 | 10 | 5 | 0.6420 | 0.6394 |
| 100 | 5 | 2 | 0.5640 | 0.5534 |
| 50 | 3 | 2 | 0.5580 | 0.5185 |

**Lý do chọn:**

1. `max_depth` là yếu tố quyết định. Nâng từ 3 lên không giới hạn đưa accuracy từ 0.558 lên 0.680 (+12,2 điểm phần trăm). Cây bị giới hạn độ sâu 3–5 không đủ sức mô tả ranh giới phi tuyến giữa ba lớp chất lượng.

2. `n_estimators` bão hòa sau khoảng 500 cây. Cấu hình 1000 cây chỉ hơn 0.004 — tương đương **2 mẫu trên 500**, nằm trong khoảng nhiễu thống kê, không phải cải thiện thật.

3. Quyết định lấy 500 cây thay vì 1000 dựa trên chi phí triển khai: `model.pkl` giảm từ **114 MB xuống 42 MB**. File này được upload lên cloud storage mỗi lần CI chạy và phải load vào RAM trên VM `e2-micro` (1 GB). Đánh đổi 0.004 accuracy để giảm 2,7 lần dung lượng là hợp lý trong bối cảnh MLOps.

**So sánh thuật toán** (Bonus 2, cùng `n_estimators=200, max_depth=5`):

| Thuật toán | accuracy |
|---|---|
| gradient_boosting | 0.6580 |
| random_forest | 0.5760 |
| logistic_regression | 0.5680 |

`logistic_regression` kém nhất vì bài toán có ranh giới phi tuyến; đã bọc `StandardScaler` nhưng vẫn không bù được. Ở cấu hình tối ưu riêng, `random_forest` (0.680) vượt `gradient_boosting` (0.658).

**Phân tích lỗi từ confusion matrix** (Bonus 3): lớp 2 (chất lượng cao) có precision cao nhất (0.771) nhưng recall thấp nhất (0.540) — mô hình bỏ sót gần một nửa rượu chất lượng cao, chủ yếu nhầm sang lớp 1. Nguyên nhân là lớp 2 chỉ chiếm 19,7% dữ liệu huấn luyện.

---

## 2. Khó khăn gặp phải và cách giải quyết

**a. Không đăng ký được tài khoản cloud.** Billing GCP báo lỗi `OR_BACR2_31` (từ chối thẻ), không thể tạo bucket và VM theo hướng dẫn gốc.

*Giải pháp:* chuyển sang **DagsHub** — miễn phí, không cần thẻ tín dụng, S3-compatible. Một dịch vụ đóng cả hai vai: DVC remote (`s3://dvc` qua endpoint `dagshub.com/<user>/<repo>.s3`) và MLflow tracking server. Đã đổi `requirements.txt` từ `dvc[gs]`/`google-cloud-storage` sang `dvc[s3]`/`boto3`, và viết `src/serve.py` hỗ trợ ba chế độ tải model: DagsHub → GCS → local fallback. Việc này đồng thời hoàn thành Bonus 1.

**b. Không đạt được ngưỡng eval 0.70 ở Bước 2.** Đã thử rất rộng nhưng trần thực sự với 2998 mẫu chỉ là ~0.686:

| Thử | accuracy tốt nhất |
|---|---|
| RandomForest (sweep cả `max_features`, `min_samples_leaf`, `class_weight`) | 0.6860 |
| ExtraTrees | 0.6820 |
| HistGradientBoosting | 0.6600 |
| GradientBoosting | 0.6020 |

*Kết luận:* giới hạn nằm ở **lượng dữ liệu**, không ở thuật toán hay siêu tham số. Khi bổ sung `train_phase2.csv` ở Bước 3 (2998 → 5996 mẫu), accuracy nhảy lên **0.746** mà không đổi một siêu tham số nào. Đây chính là điều lab muốn minh chứng: ở Bước 2 eval gate **chặn** deploy đúng như thiết kế, và chỉ dữ liệu mới mới mở được cổng đó — lý do tồn tại của continuous training.

**c. Unit test fail không ổn định (`MissingConfigException`).** Test pass khi chạy riêng nhưng fail sau khi chạy `python src/train.py`.

*Nguyên nhân:* với `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`, MLflow vẫn tạo `./mlruns` để chứa artifact nhưng **không** ghi `mlruns/0/meta.yaml` (metadata nằm trong SQLite). Lần chạy pytest sau đó không có biến môi trường này nên MLflow đọc `./mlruns` như file store, thấy thiếu `meta.yaml` và crash.

*Giải pháp:* thêm `tests/conftest.py` với fixture `autouse` trỏ tracking URI vào một thư mục tạm **chưa tồn tại** — điều kiện để `FileStore` bootstrap experiment mặc định. Test giờ chạy giống nhau trên máy cá nhân và trên CI runner, không phụ thuộc biến môi trường hay thư mục sót lại.

**d. Bước "Download previous metrics" đọc ra rỗng (Bonus 4).** Ban đầu tôi `echo` giá trị output ngay trong chính step đặt nó. GitHub Actions đánh giá biểu thức `${{ }}` **trước** khi step chạy, nên luôn ra rỗng. Đã chuyển phần thông báo vào trong Python và ghi ra `stderr`, còn `stdout` dành riêng cho `$GITHUB_OUTPUT`.

---

## 3. Tóm tắt kết quả

| Hạng mục | Trạng thái |
|---|---|
| MLflow tracking, 20+ run | Đạt |
| Unit test (3 test) | Pass, ổn định trong 3 môi trường |
| DVC remote (DagsHub) | Đã cấu hình |
| CI/CD 4 jobs | Test → Train → Eval → Deploy |
| Eval gate | Chặn đúng ở 0.68 < 0.70 |
| Bonus 1–5 | Hoàn thành |
