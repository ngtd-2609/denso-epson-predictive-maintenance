# Bản đóng gói GitHub — 2026-10-04

Phiên bản đóng gói: `EPSON_PERSON1_GITHUB_V1`. Giao thức dữ liệu vẫn là `EPSON_PERSON1_HANDOFF_V1`/`prepared_v1`; không đổi split, k, scaler, window hoặc nhãn.

## Thay đổi

- Khai báo pandas cho loader; tách dependency QA khỏi runtime.
- Chọn lọc code, metadata và báo cáo để commit; raw, NPZ, archive, ảnh QA đầy đủ, cache và bản tái tạo dư giữ ngoài Git.
- Thêm `handoff.py` chạy tương đối theo vị trí repo, rebasing đường dẫn raw từ recording ID trong bộ split đã chốt.
- Thêm checksum nội dung mảng để xác nhận 12 file NPZ tái tạo đúng dữ liệu chuẩn.
- Manifest phát hành mới bao phủ file thực sự trong repo (trừ chính manifest), trong khi các manifest QA cũ được giữ như hồ sơ lịch sử. Một số checksum code/dependency lịch sử có thể không còn khớp bản đóng gói mới; không sửa hồ sơ cũ để giả lập một lần QA chưa chạy.

## Cách hiểu bằng chứng

`results/qa_v2` ghi lại lần chạy Người 1. Các script QA cũ được lưu để review/tái hiện có chủ đích, có thể chứa đường dẫn máy gốc và ghi đè báo cáo; không phải lệnh onboarding. Đầu vào chuẩn cho người nhận là README gốc, `handoff.py`, split và metadata đã chốt.

Manifest công khai chứa tên file, thống kê và checksum, không chứa chuỗi tín hiệu thô. Dataset không được cấp lại giấy phép bởi repo này; giữ nguồn Epson và kiểm tra điều kiện sử dụng cho phạm vi dự kiến.

## Nghiệm thu tối thiểu

1. Clone repo ở đường dẫn của người nhận, ghi commit (`git rev-parse HEAD`).
2. Cài requirements trong môi trường riêng và chạy `handoff.py verify`.
3. Lấy raw từ nguồn chính thức, đối chiếu archive hash, extract, restore; hoặc nhận NPZ đúng phiên bản qua kênh được phép.
4. Chạy `handoff.py verify --data` và `handoff.py smoke`.
5. Người 2/3 kiểm tra giao diện và xác nhận quy tắc train/validation/test. Người 4 xác nhận phạm vi tuyên bố.

Không cần chạy lại near-duplicate toàn bộ, tạo lại ảnh QA, hoặc train mô hình để nghiệm thu Person 1. Không đưa `repro_release_v2` lên Git vì nội dung mảng đã được báo cáo trùng bản chuẩn.

## Kiểm tra bản đóng gói

- 9/9 tests phần mềm đạt (7 tests pipeline và 2 tests bàn giao).
- Kiểm tra nội dung tất cả 12 NPZ gốc khớp manifest mảng.
- Bản sao sạch của các file được chọn cho Git, đặt ở thư mục khác và ban đầu không có NPZ: `restore` từ raw Epson thành công; cả 12 NPZ khớp nội dung mảng chuẩn, report/scaler/metadata cửa sổ khớp.
- Smoke test trên dữ liệu vừa tái tạo đạt cả normal_only, k1, k2, k4.
- Môi trường venv mới chỉ cài `requirements.txt`: kiểm tra release và smoke test tại bản sao khác đường dẫn đều đạt. Không phụ thuộc pandas/PyTorch có sẵn trong môi trường phát triển.
- Đây là kiểm tra trên cùng máy Windows, không phải xác nhận độc lập từ thành viên khác hay kiểm tra Linux/macOS. Không train mô hình trong quá trình đóng gói.
