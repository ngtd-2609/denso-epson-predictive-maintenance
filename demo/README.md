# Demo sinh tín hiệu rung

Demo dùng cVAE đã huấn luyện ở thiết lập k1, seed 11 và chạy trên CPU. Mỗi mẫu gồm XYZ dài một giây, lấy mẫu 3.000 Hz. Có thể chọn nhãn lỗi 1–5 và sinh 1–100 mẫu.

Từ thư mục gốc repo trên Windows:

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r demo/requirements.txt
.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
powershell -NoProfile -ExecutionPolicy Bypass -File demo/chay_demo.ps1 -Lop 1 -SoMau 5 -Seed 2026
```

Linux/macOS dùng `.venv/bin/python demo/demo.py --class-id 1 --samples 5 --seed 2026 --out demo/ket_qua_moi` sau khi cài cùng thư viện.

Chương trình xuất `synthetic_samples.npz`, `first_sample.csv` và `generation_report.json`. NPZ chứa tín hiệu chuẩn hóa, tín hiệu đổi về mm/s, nhãn và cờ mẫu tổng hợp. Thư mục `vi_du` có năm mẫu lớp 1 đã sinh bằng seed 2026.

`RMS ratio XYZ` so sánh biên độ mẫu sinh với trung bình dữ liệu huấn luyện. `WARNING: True` nghĩa là có kênh nằm ngoài khoảng chẩn đoán [0,5; 2]. Mô hình hiện tại sinh tín hiệu có biên độ thấp, nên cần đọc cảnh báo cùng kết quả.

Bảng cuối màn hình là kết quả test đã lưu của thí nghiệm, không phải đánh giá mới cho các mẫu vừa sinh. Không cần GPU hoặc huấn luyện lại để chạy demo.
