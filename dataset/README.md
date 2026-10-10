# Dữ liệu và quy trình chuẩn bị

Nguồn: [Epson Rotor Kit Vibration Dataset](https://www.epsondevice.com/sensing/en/dataset/index.html), bản trimmed `rotorkit_dataset_trimmed.zip`.

Dữ liệu gồm 96 bản ghi vận tốc rung XYZ (mm/s), 3.000 Hz, tốc độ danh định 1.200 rpm. Sáu lớp gồm một bình thường và năm cấu hình mất cân bằng. Tập train/validation/test có 48/24/24 bản ghi. Các cửa sổ dài 3.000 điểm và không chồng lấn.

Đặt archive tải từ Epson vào `dataset/downloads/rotorkit_dataset_trimmed.zip`. SHA-256 của archive dùng trong thử nghiệm:

```text
aff2b544ad7c3c4c219cb35414c3b0560012af8b3da01854da69bf355fc884c8
```

Chạy từ thư mục gốc repo, sau khi tạo môi trường Python:

```powershell
.venv/Scripts/python.exe -m pip install -r dataset/requirements.txt
.venv/Scripts/python.exe dataset/prepare_data.py verify
.venv/Scripts/python.exe dataset/extract_archive.py --archive dataset/downloads/rotorkit_dataset_trimmed.zip --out dataset/data/raw_epson
.venv/Scripts/python.exe dataset/prepare_data.py restore --raw dataset/data/raw_epson
.venv/Scripts/python.exe dataset/prepare_data.py verify --data
.venv/Scripts/python.exe dataset/prepare_data.py smoke
```

Quy trình tái tạo 12 file NPZ theo split, scaler và cửa sổ đã chốt, rồi so sánh nội dung từng mảng với checksum chuẩn. Không ghi đè NPZ đã tồn tại. Tín hiệu thật tải trực tiếp từ Epson; repo chứa metadata và mã chuẩn bị.

```python
from dataset.loader_v1 import EpsonPreparedLoader
loader = EpsonPreparedLoader()
train = loader.load('k1', 'train')
print(train.X.shape)  # (1222, 3, 3000)
```

Mức k1/k4 dùng một/bốn bản ghi lỗi mỗi lớp, cùng tám bản ghi bình thường. Scaler chỉ học từ dữ liệu train được phép dùng. Validation chọn mô hình/ngưỡng, test đánh giá cuối. Giao thức và nhãn nằm trong `results/qa_v2`.

Tham khảo dữ liệu: H. Nagahama, K. Inoue, M. Todorokihara, M. Yoshioka, “Empirical Investigation of the Impact of Phase Information on Fault Diagnosis of Rotating Machinery”, arXiv:2512.15344, 2025. Dữ liệu Epson thuộc quyền của chủ sở hữu tương ứng.
