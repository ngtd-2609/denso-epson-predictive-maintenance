# Kết quả kiểm chứng

`Ket_qua_tong_hop.json` chứa trung bình ba seed của độ chính xác nhị phân, tỷ lệ phát hiện lỗi và tỷ lệ báo nhầm theo bản ghi. Số liệu chi tiết nằm trong `thuc_nghiem/report/metrics.csv`.

`thuc_nghiem/k1` và `thuc_nghiem/k4` lưu trọng số, ngưỡng, đường huấn luyện, dự đoán validation/test và ma trận nhầm lẫn cho từng phương án, seed. `report/synthetic_quality.json` lưu chẩn đoán mẫu cVAE. `FINAL_LOCK.json` và `protocol.json` ghi cấu hình và checksum trước đánh giá test. `source_snapshot` là mã gốc dùng trong lần chạy.

Các file bằng chứng được giữ nguyên nội dung. Đường dẫn trong một số JSON ghi môi trường lúc chạy; không cần dùng đường dẫn đó để xem số liệu hoặc chạy demo. Ba seed đánh giá cùng 24 bản ghi test, không tương đương 72 bản ghi độc lập.
