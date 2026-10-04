EPSON — BÀN GIAO PHẦN NGƯỜI 1

Hướng dẫn đầy đủ:
D:\DENSO_FACTORY\output\bo_tai_lieu_nhom\04_HUONG_DAN_NGUOI_1_EPSON.txt

Kết quả lần chạy:
- 96 tệp hợp lệ, 16 mỗi lớp.
- Không có bản ghi trùng y hệt theo byte hoặc ma trận số.
- Split 48 train pool / 24 validation / 24 test, seed42.
- Ngân sách lỗi k1/k2/k4 lồng nhau, seed101.
- Cửa sổ đồng bộ [3,3000], 1 giây, không chồng lấn.
- Scaler fit từ đúng train từng ngân sách; normal-only riêng.
- 7 unit/integration tests bằng fixture giả đã pass.
- Verification dữ liệu thật pass cả bốn profile; không có cửa sổ chuẩn hóa trùng y hệt giữa các split.

Điểm bắt đầu cho người 2/3:
results/prepared_v1/preparation_report.json
results/prepared_v1/verification.json
results/prepared_v1/k1/train.npz

validation/test của k1/k2/k4 dùng cùng tệp và cùng vị trí cửa sổ,
nhưng giá trị X khác vì scaler từng k khác nhau. Không trộn các profile.
Chưa train mô hình. Không có đánh giá hiệu quả phát hiện lỗi.

Giới hạn: không biết session IDs; không có timestamp từng mẫu;
gần trùng/dịch pha chưa loại trừ đầy đủ; clipping chưa có ngưỡng cảm biến xác minh;
cùng hệ thử và tốc độ. Giấy phép thương mại/tái phân phối cần làm rõ.

Phạm vi kiểm tra test hiện tại chỉ là cấu trúc, checksum, tính hữu hạn,
nhãn/ID và trùng y hệt; không dùng để chọn mô hình hay ngưỡng.

Nguồn:
https://www.epsondevice.com/sensing/en/dataset/index.html
https://arxiv.org/html/2512.15344v1 (IV-A mô tả các tệp độc lập)

Code đã kiểm tra trên Python 3.13.5, NumPy 2.3.5, pytest 9.1.1.
Không cần GPU/PyTorch cho phần dữ liệu. requirements.txt là phiên bản đã chạy,
không phải chỉ thị nâng cấp môi trường hiện có.

Kiểm thử từ D:\DENSO_FACTORY:
python -m pytest person1_data/tests -q -p no:cacheprovider

Tái chạy dữ liệu phải dùng thư mục output mới; không ghi đè kết quả cũ.
Xem tài liệu 04 để lấy đầy đủ lệnh extract/audit/split/prepare/verify.
Thư mục gốc hiện chưa là Git repository; không tự tạo commit giả.
verification.json có SHA256 của code data_pipeline.py được sử dụng.
