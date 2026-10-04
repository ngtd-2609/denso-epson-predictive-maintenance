# DENSO — Epson: bàn giao dữ liệu Người 1

Gói dữ liệu phục vụ thí nghiệm sinh tín hiệu bất thường và đánh giá detector khi thiếu dữ liệu lỗi. **Chưa có mô hình đã train hoặc kết quả cải thiện detector.** Tên repo không có nghĩa dữ liệu hỗ trợ dự đoán thời điểm hỏng/RUL.

## Bắt đầu nhanh

Các lệnh dưới đây chạy từ thư mục gốc của repo, không phụ thuộc ổ đĩa hay tên tài khoản. Môi trường đã kiểm tra: Python 3.13.5. GPU không cần cho bước này.

```powershell
git clone https://github.com/ngtd-2609/denso-epson-predictive-maintenance.git
cd denso-epson-predictive-maintenance
python -m venv .venv
# Windows PowerShell: dùng trực tiếp Python trong venv, không cần đổi execution policy.
.venv/Scripts/python.exe -m pip install -r person1_data/requirements.txt
.venv/Scripts/python.exe person1_data/handoff.py verify
```

Linux/macOS dùng `.venv/bin/python` thay cho `.venv/Scripts/python.exe`. Nếu đã có môi trường phù hợp, có thể dùng `python` trong các lệnh.

## Nhận dữ liệu

Repo chứa code và metadata, **không chứa raw CSV, archive hoặc mảng NPZ**. Lấy bản **trimmed** tại [trang Epson chính thức](https://www.epsondevice.com/sensing/en/dataset/index.html), tên archive `rotorkit_dataset_trimmed.zip`. Đọc điều kiện sử dụng của nguồn; quyền tái phân phối/chuyển sang phạm vi thương mại chưa được xác minh trong hồ sơ này.

Đặt archive tại `person1_data/downloads/rotorkit_dataset_trimmed.zip` (tạo thư mục nếu chưa có). SHA-256 của bản đã dùng:

```text
aff2b544ad7c3c4c219cb35414c3b0560012af8b3da01854da69bf355fc884c8
```

```powershell
Get-FileHash person1_data/downloads/rotorkit_dataset_trimmed.zip -Algorithm SHA256
.venv/Scripts/python.exe person1_data/extract_archive.py --archive person1_data/downloads/rotorkit_dataset_trimmed.zip --out person1_data/data/raw_epson
.venv/Scripts/python.exe person1_data/handoff.py restore --raw person1_data/data/raw_epson
.venv/Scripts/python.exe person1_data/handoff.py verify --data
.venv/Scripts/python.exe person1_data/handoff.py smoke
```

Chỉ tiếp tục nếu checksum archive khớp. Extract và restore từ chối ghi đè. Nếu đã có raw/NPZ đúng phiên bản, bỏ qua bước tương ứng và chạy kiểm tra. Nếu restore bị gián đoạn, giữ bản cũ để kiểm tra hoặc dùng một checkout sạch; không tự xóa dữ liệu chưa rõ nguồn.

`restore` dùng **96 recording và split đã chốt**, xác minh checksum raw, tái tạo bốn profile rồi đối chiếu report, scaler, metadata cửa sổ và nội dung từng mảng với bản chuẩn trước khi công bố NPZ. Không chia lại dữ liệu. Chuẩn bị vài GB dung lượng trống cho raw, đầu ra và vùng tạm.

Nếu sử dụng một gói prepared được chia sẻ hợp lệ qua kênh khác, đặt 12 file vào `person1_data/results/prepared_v1/{normal_only,k1,k2,k4}/{train,validation,test}.npz`, sau đó chạy `verify --data` và `smoke`. Gói phải khớp manifest; không dùng chỉ vì tên file giống nhau.

## Giao diện cho Người 2 và Người 3

Từ thư mục `person1_data`:

```python
from loader_v1 import EpsonPreparedLoader

loader = EpsonPreparedLoader()
train = loader.load("k1", "train")
scaler = loader.get_scaler("k1")
print(train.X.shape)  # (1222, 3, 3000)
```

| Profile | Cửa sổ train | Recording train | Scaler fit trên |
|---|---:|---:|---|
| normal_only | 752 | 8 | 8 recording normal train |
| k1 | 1222 | 13 | 8 normal + 1 recording/lớp bất thường |
| k2 | 1692 | 18 | 8 normal + 2 recording/lớp bất thường |
| k4 | 2632 | 28 | 8 normal + 4 recording/lớp bất thường |

Mỗi profile có 2.256 cửa sổ validation và 2.256 cửa sổ test. Các profile dùng cùng recording/vị trí cửa sổ đánh giá nhưng **giá trị chuẩn hóa khác nhau** do scaler khác nhau.

- `X`: float32 `[N, 3, 3000]`, XYZ đồng bộ, z-score; raw là vận tốc rung mm/s, 3.000 Hz.
- `y_class`: 0..5 cấu hình; `y_binary`: 0 normal, 1 abnormal. Đây là năm cấu hình mất cân bằng, không phải năm cơ chế hỏng độc lập.
- `recording_id`, `start_sample`: truy xuất nguồn và đánh giá theo recording. Các cửa sổ không phải các phiên đo độc lập.
- Người 2: detector/baseline, chọn ngưỡng bằng validation; test chỉ đánh giá cuối.
- Người 3: generator chỉ dùng train đúng k, sinh XYZ đồng thời, lưu scaler/provenance; không dùng checkpoint học từ ngân sách lớn hơn cho k nhỏ.
- Mọi phương pháp so sánh trong cùng k phải dùng cùng dữ liệu thực, scaler và tập đánh giá.
- PyTorch là tùy chọn riêng cho training/adapter; cài bản phù hợp GPU/môi trường của người nhận. Không cài nguyên `pip_freeze.txt` của máy Người 1.

## Tài liệu và bằng chứng

- [Hợp đồng dữ liệu](person1_data/results/qa_v2/data_contract.json), [protocol](person1_data/results/qa_v2/protocol_frozen.txt), [quyền truy cập và scarcity](person1_data/results/qa_v2/access_and_scarcity.csv).
- [Dataset card](person1_data/results/qa_v2/dataset_card.txt), [nguồn](person1_data/results/qa_v2/source_register.csv).
- [Bàn giao theo vai trò](person1_data/results/qa_v2/HANDOFF_README.txt), [mẫu xác nhận người nhận](person1_data/results/qa_v2/RECIPIENT_ACCEPTANCE_TEMPLATE.txt).
- [Thay đổi đóng gói và giới hạn](docs/HANDOFF_RELEASE.md).

Các báo cáo cũ và script QA giữ nguyên để truy vết lần chạy ban đầu; có thể chứa đường dẫn `D:\DENSO_FACTORY` và checksum phiên bản cũ. **Quy trình nhận bàn giao hiện hành là README này và `handoff.py`**, không chạy lại script stage4/stage6/stage7 để nhận dữ liệu. Manifest hiện hành là `release_manifest.json`; `prepared_arrays_manifest.json` xác minh nội dung NPZ, không phụ thuộc metadata container ZIP.

## Kiểm thử phần mềm

```powershell
.venv/Scripts/python.exe -m pip install -r person1_data/requirements-qa.txt
.venv/Scripts/python.exe -m pytest person1_data/tests -q -p no:cacheprovider
```

Tests dùng fixture giả để kiểm tra phần mềm, không phải bằng chứng hiệu quả phát hiện bất thường.

## Giới hạn và nghiệm thu

Một hệ thử, một tốc độ danh nghĩa 1.200 rpm; không có session ID/timestamp từng mẫu. Kiểm tra gần trùng không chứng minh độc lập phiên đo hoàn toàn. Không tuyên bố dữ liệu đo tại DENSO, tổng quát đa tốc độ hoặc dự báo thời điểm hỏng.

Người nhận phải tự chạy kiểm tra trên máy mình và ghi commit, môi trường, kết quả cùng giới hạn còn mở vào biên bản nhận bàn giao. Kiểm tra của Người 1/AI không thay thế xác nhận đó. Khi thay split/scaler/window/quy trình, tạo phiên bản dữ liệu mới; khi chỉ sửa code/tài liệu, cập nhật manifest phát hành code tương ứng.
