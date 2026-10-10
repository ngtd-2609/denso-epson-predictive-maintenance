# Mã nguồn thí nghiệm

`models.py` định nghĩa cVAE và CNN một chiều. `core.py` chứa phép tăng cường, tính chỉ số và chọn ngưỡng. `training.py` huấn luyện và đánh giá; `pipeline.py` quản lý các bước chạy. Cấu hình báo cáo là `configs/final.json`: k1/k4, seed 11/22/33, 1.600 bước bộ sinh và 800 bước bộ phát hiện.

Môi trường đã dùng: Python 3.13.5, PyTorch 2.7.1, GPU RTX 3050 Laptop 4 GB. Cài thư viện bằng `requirements-verified.txt`, cài PyTorch riêng theo CPU/GPU. Chuẩn bị dữ liệu theo `dataset/README.md`.

Chạy từ thư mục gốc repo, dùng một thư mục kết quả mới:

```powershell
.venv/Scripts/python.exe -m experiment doctor
.venv/Scripts/python.exe -m experiment train --config experiment/configs/final.json --run output/thuc_nghiem_moi --device cuda
.venv/Scripts/python.exe -m experiment freeze --run output/thuc_nghiem_moi
.venv/Scripts/python.exe -m experiment evaluate --run output/thuc_nghiem_moi --device cuda
.venv/Scripts/python.exe -m experiment report --run output/thuc_nghiem_moi
```

Chốt cấu hình sau khi xem validation và trước khi mở test. Tập test không dùng để chọn lại mô hình hoặc ngưỡng.

`ket_qua/thuc_nghiem/source_snapshot` lưu mã gốc tại thời điểm chạy số liệu trong báo cáo. Mã chạy ở thư mục `experiment` dùng module dữ liệu tên `dataset`; các phép tính, kiến trúc và cấu hình giữ như mã gốc. Checkpoint, dự đoán và khóa thí nghiệm trong `ket_qua/thuc_nghiem` là bằng chứng của lần chạy đã hoàn thành.

Kiểm tra phần mềm:

```powershell
.venv/Scripts/python.exe -m pytest experiment/tests dataset/tests -q -p no:cacheprovider
```
