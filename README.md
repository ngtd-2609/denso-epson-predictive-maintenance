# Tạo dữ liệu rung bất thường để hỗ trợ phát hiện lỗi thiết bị

**Nguyen Tung Duong — Trường Phenikaa**

Thử nghiệm bổ sung dữ liệu rung bằng cVAE khi chỉ có một hoặc bốn bản ghi lỗi mỗi lớp để huấn luyện. Bốn phương án gồm dữ liệu thật, lặp mẫu, tăng cường đơn giản và cVAE. Tất cả dùng cùng CNN và đánh giá trên dữ liệu thật giữ riêng của Epson.

## Hồ sơ gửi Ban tổ chức

| Nội dung | File hoặc thư mục |
|---|---|
| Báo cáo bài toán, phương pháp và kết quả | [bao_cao/Bao_cao.pdf](bao_cao/Bao_cao.pdf) |
| Slide thuyết trình | [thuyet_trinh/Thuyet_trinh.pptx](thuyet_trinh/Thuyet_trinh.pptx) |
| Bản PDF của slide | [thuyet_trinh/Thuyet_trinh.pdf](thuyet_trinh/Thuyet_trinh.pdf) |
| Bộ sinh đã huấn luyện và demo CPU | [demo](demo/README.md) |
| Dữ liệu tăng cường 4.512 đoạn | [du_lieu](du_lieu/README.md) |
| Mã chuẩn bị dữ liệu và thí nghiệm | [dataset](dataset/README.md), [experiment](experiment/README.md) |
| Số liệu, trọng số và dự đoán kiểm chứng | [ket_qua](ket_qua/README.md) |

Tải [gói hồ sơ tại Releases](https://github.com/ngtd-2609/denso-epson-predictive-maintenance/releases/tag/btc-submission-2026-10-11) nếu cần nộp một file ZIP. Gói nhẹ chứa tài liệu, demo, mã nguồn và bằng chứng; gói đầy đủ có thêm bộ dữ liệu tăng cường.

## Kết quả chính

Trung bình ba seed 11/22/33 trên cùng 24 bản ghi test; các chỉ số dưới đây tính theo phần trăm. k1/k4 tương ứng một/bốn bản ghi lỗi mỗi lớp để huấn luyện, cùng tám bản ghi bình thường.

| Mức | Phương án | Độ chính xác | Phát hiện lỗi | Báo nhầm |
|---|---|---:|---:|---:|
| k1 | Chỉ dùng dữ liệu thật | 25.0 | 11.7 | 8.3 |
| k1 | Lặp lại mẫu thật | 30.6 | 20.0 | 16.7 |
| k1 | Tăng cường đơn giản | 100.0 | 100.0 | 0.0 |
| k1 | Sinh bằng cVAE | 50.0 | 43.3 | 16.7 |
| k4 | Chỉ dùng dữ liệu thật | 95.8 | 96.7 | 8.3 |
| k4 | Lặp lại mẫu thật | 93.1 | 91.7 | 0.0 |
| k4 | Tăng cường đơn giản | 100.0 | 100.0 | 0.0 |
| k4 | Sinh bằng cVAE | 81.9 | 78.3 | 0.0 |

cVAE tăng tỷ lệ phát hiện ở k1 nhưng cũng tăng báo nhầm; ở k4, tỷ lệ phát hiện thấp hơn chỉ dùng dữ liệu thật. Tăng cường đơn giản cho kết quả tốt nhất trong phép so sánh này. RMS mẫu cVAE chỉ khoảng 17,5–25,8% so với dữ liệu thật theo kênh trong các tổ hợp đã kiểm tra.

Kết quả 100% ở bảng là phân biệt bình thường/bất thường theo bản ghi trên tập test nhỏ, không phải mọi đoạn một giây đều đúng. Tập test chỉ có bốn bản ghi bình thường. Dữ liệu gồm một hệ thử và một tốc độ danh định, chưa chứng minh hiệu quả trên thiết bị khác hoặc lỗi chưa xuất hiện trong huấn luyện.

## Chạy demo

Theo [hướng dẫn demo](demo/README.md), cài Python và PyTorch CPU rồi chạy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File demo/chay_demo.ps1 -Lop 1 -SoMau 5 -Seed 2026
```

Demo xuất tín hiệu XYZ mới và chẩn đoán biên độ, không cần GPU. Bảng chỉ số hiển thị là kết quả thí nghiệm đã lưu.

## Áp dụng cho dữ liệu khác

Có thể giữ quy trình so sánh và dùng cVAE sinh dữ liệu theo nhãn. Khi đổi bộ dữ liệu, cần điều chỉnh cách đọc, số kênh, số lớp, tần số lấy mẫu và độ dài cửa sổ, sau đó huấn luyện và đánh giá lại. Mã hiện tại dành cho Epson (3 kênh, 3.000 điểm, 6 lớp); chưa hỗ trợ thay mọi bộ dữ liệu tự động.

## Nguồn và bản quyền

Dữ liệu thật: [Epson Rotor Kit Vibration Dataset](https://www.epsondevice.com/sensing/en/dataset/index.html). Phương pháp tham khảo [Kingma và Welling](https://arxiv.org/abs/1312.6114), [Sohn và cộng sự](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html). Bài toán do BTC DENSO cung cấp.

Copyright © 2026 Nguyen Tung Duong. All rights reserved. Xem [LICENSE](LICENSE). Dữ liệu và tài liệu bên thứ ba thuộc quyền của chủ sở hữu tương ứng.
