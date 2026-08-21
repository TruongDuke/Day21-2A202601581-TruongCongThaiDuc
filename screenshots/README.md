# Ảnh chụp minh chứng

Đặt ảnh vào thư mục này với **đúng tên file** dưới đây (định dạng `.png`).
`BAOCAO.md` đã nhúng sẵn các đường dẫn này, chỉ cần copy ảnh vào là hiện.

| Tên file | Nội dung cần chụp |
|---|---|
| `01-mlflow-ui.png` | MLflow UI, bảng 22 run sort theo accuracy giảm dần, thấy đủ cột accuracy + f1_score + siêu tham số, kèm dòng "22 matching runs" |
| `02-actions-buoc2-eval-chan.png` | Tab Actions, run Bước 2 — Test/Train xanh, **Eval đỏ**, Deploy skipped |
| `03-actions-buoc3-4-jobs-xanh.png` | Tab Actions, run Bước 3 — cả 4 jobs xanh |
| `04-curl-health-predict.png` | Terminal chạy `curl /health` và `curl /predict` |
| `05-dvc-storage.png` | Thư mục `dvcstore/` trên GitHub + trang Artifacts của run (thay cho Cloud Storage Console) |

## Link trực tiếp đến các run

- Bước 2 (eval gate chặn): https://github.com/TruongDuke/Day21-2A202601581-TruongCongThaiDuc/actions/runs/32456187742
- Bước 3 (4 jobs xanh): https://github.com/TruongDuke/Day21-2A202601581-TruongCongThaiDuc/actions/runs/32456566223
