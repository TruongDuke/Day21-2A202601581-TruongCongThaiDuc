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

**a. Không đăng ký được tài khoản cloud.** Billing GCP báo lỗi `OR_BACR2_31` (từ chối thẻ) — thẻ bị từ chối ở bước xác thực thanh toán, không thể tạo bucket và VM theo hướng dẫn gốc. AWS và Azure đều cũng yêu cầu thẻ tín dụng.

*Giải pháp:* dựng lại toàn bộ pipeline **chỉ trên GitHub**, không cần đăng ký thêm dịch vụ nào:

| Thành phần | Bản gốc | Thay thế |
|---|---|---|
| DVC remote | GCS bucket | `./dvcstore` trong repo (392 KB) |
| Model storage | `gs://.../models/latest/` | GitHub Actions artifacts |
| Metrics giữa các lần chạy | cloud storage | GitHub Actions cache |
| VM serving | GCE + systemd + SSH | FastAPI khởi động trên runner |

Quy trình DVC vẫn nguyên vẹn: `dvc add` → `dvc push` → `dvc pull` trong CI đều chạy thật, chỉ khác là remote nằm trong repo thay vì trên cloud. `src/serve.py` được viết hỗ trợ ba chế độ tải model (DagsHub S3 → GCS → local) nên chỉ cần đặt biến môi trường là chuyển sang cloud được, không phải sửa code.

*Hạn chế cần nêu rõ:* rubric yêu cầu "dữ liệu hiển thị trên cloud storage" và "VM trả về kết quả tại endpoint". Phương án này chứng minh đầy đủ về mặt chức năng nhưng không dùng cloud thật, nên hai hạng mục đó (24 điểm) cần giảng viên xác nhận cách chấm.

**b. Không đạt được ngưỡng eval 0.70 ở Bước 2.** Đã thử rất rộng nhưng trần thực sự với 2998 mẫu chỉ là ~0.686:

| Thử | accuracy tốt nhất |
|---|---|
| RandomForest (sweep cả `max_features`, `min_samples_leaf`, `class_weight`) | 0.6860 |
| ExtraTrees | 0.6820 |
| HistGradientBoosting | 0.6600 |
| GradientBoosting | 0.6020 |

*Kết luận:* giới hạn nằm ở **lượng dữ liệu**, không ở thuật toán hay siêu tham số. Khi bổ sung `train_phase2.csv` ở Bước 3 (2998 → 5996 mẫu), accuracy nhảy lên **0.7500** mà không đổi một siêu tham số nào. Đây chính là điều lab muốn minh chứng: ở Bước 2 eval gate **chặn** deploy đúng như thiết kế, và chỉ dữ liệu mới mới mở được cổng đó — lý do tồn tại của continuous training.

(Đo cục bộ trên macOS cho 0.7460, CI trên Ubuntu cho 0.7500 — lệch 2 mẫu trên 500 do khác nền tảng và thứ tự luồng trong RandomForest.)

**c. Unit test fail không ổn định (`MissingConfigException`).** Test pass khi chạy riêng nhưng fail sau khi chạy `python src/train.py`.

*Nguyên nhân:* với `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`, MLflow vẫn tạo `./mlruns` để chứa artifact nhưng **không** ghi `mlruns/0/meta.yaml` (metadata nằm trong SQLite). Lần chạy pytest sau đó không có biến môi trường này nên MLflow đọc `./mlruns` như file store, thấy thiếu `meta.yaml` và crash.

*Giải pháp:* thêm `tests/conftest.py` với fixture `autouse` trỏ tracking URI vào một thư mục tạm **chưa tồn tại** — điều kiện để `FileStore` bootstrap experiment mặc định. Test giờ chạy giống nhau trên máy cá nhân và trên CI runner, không phụ thuộc biến môi trường hay thư mục sót lại.

**d. Bước đọc metrics cũ luôn ra rỗng (Bonus 4).** Trong phiên bản đầu, tôi `echo` giá trị output ngay trong chính step đặt nó. GitHub Actions đánh giá biểu thức `${{ }}` **trước** khi step chạy, nên biến luôn rỗng bất kể logic bên trong đúng hay sai. Đã tách phần thông báo ra khỏi `$GITHUB_OUTPUT` và chỉ đọc output đó ở step sau.

**e. Health check "pass" giả do trùng port.** Khi test đường serving trên máy cá nhân, `curl /health` trả về `200 {"status":"ok","env":"development"}` — nhưng `src/serve.py` không hề trả về field `env`. Kiểm tra `lsof` thì thấy một process khác đã chiếm port 8000 từ trước; uvicorn của tôi chết ngay với `address already in use` còn `curl` thì trúng service kia.

*Bài học:* health check chỉ dựa vào mã HTTP 200 là không đủ — nó xác nhận "có ai đó đang lắng nghe", không phải "service của tôi đang chạy". Đã thêm bước kiểm tra process của chính mình còn sống (`kill -0 $PID`) trước và trong vòng lặp retry, và in log uvicorn khi thất bại. Trên CI runner port luôn trống nên lỗi này không xảy ra, nhưng nếu deploy lên VM dùng chung thì đây đúng là tình huống gây nhầm lẫn nguy hiểm: pipeline báo xanh trong khi model mới chưa hề được load.

---

## 3. Bằng chứng pipeline chạy thật

Hai lần chạy trên GitHub Actions minh chứng đầy đủ vòng continuous training:

**Bước 2** — [run 32456187742](https://github.com/TruongDuke/Day21-2A202601581-TruongCongThaiDuc/actions/runs/32456187742), huấn luyện trên 2998 mẫu:

| Job | Kết quả |
|---|---|
| Unit Test | success |
| Train | success — accuracy 0.6800 |
| Eval | **failure** — `FAILED: huy deploy vi accuracy 0.6800 < nguong 0.7` |
| Deploy | skipped |

**Bước 3** — [run 32456566223](https://github.com/TruongDuke/Day21-2A202601581-TruongCongThaiDuc/actions/runs/32456566223), kích hoạt tự động bởi một `git push` chứa con trỏ DVC mới:

| Job | Kết quả |
|---|---|
| Unit Test | success |
| Train | success — `dvc pull` lấy 5996 mẫu, accuracy 0.7500 |
| Eval | success — `Accuracy cu: 0.6800 (thay doi: +0.0700)` → PASSED |
| Deploy | success — `{"status":"ok"}`, `/predict` trả nhãn hợp lệ, 3 đặc trưng → HTTP 400 |

Bonus 4 được xác nhận chạy thật: job Eval của Bước 3 đọc được `PREV_ACC: 0.68` từ cache của lần chạy trước và so sánh trước khi cho deploy.

---

## 4. Tóm tắt kết quả

| Hạng mục | Trạng thái |
|---|---|
| MLflow tracking, 20+ run | Đạt |
| Unit test (3 test) | Pass, ổn định trong 3 môi trường |
| DVC remote (`./dvcstore`) | Đã cấu hình, `dvc pull` chạy trong CI |
| CI/CD 4 jobs | Test → Train → Eval → Deploy, không cần tài khoản cloud |
| Eval gate | Chặn đúng ở 0.68 < 0.70, mở ở 0.75 |
| Bonus 1–5 | Hoàn thành (Bonus 4 xác nhận trên CI) |
| Bước 3 - tự động hóa | 1 push kích hoạt cả 4 jobs, không tác động thủ công |
