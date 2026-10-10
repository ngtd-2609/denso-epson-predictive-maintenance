# Bộ dữ liệu tăng cường

Tải `augmented_k1_seed11.npz` tại [bản phát hành hồ sơ BTC](https://github.com/ngtd-2609/denso-epson-predictive-maintenance/releases/tag/btc-submission-2026-10-11). File có 4.512 đoạn: 1.222 đoạn thật thuộc tập train k1 và 3.290 đoạn sinh bằng cVAE seed 11. Mỗi lớp có 752 đoạn sau bổ sung. Dữ liệu này chỉ dùng cho huấn luyện.

File chứa `X` chuẩn hóa, `X_raw_mm_s` theo mm/s, `y_class`, `y_binary`, `is_added`, `source_recording_id`, `start_sample` và `synthetic_id`. `is_added=True` đánh dấu mẫu sinh. Mẫu bổ sung không phải bản ghi thật độc lập.

SHA-256:

```text
1b943a1ed2af548e1a1bca889c8f1d3bcc89a89e811a403bdc7953c5f018d049
```

Metadata, scaler và nguồn các bản ghi train nằm trong JSON đi kèm. Dữ liệu thật có nguồn Epson; dữ liệu tổng hợp do mô hình tạo ra. Đọc cùng kết quả chất lượng tín hiệu trong báo cáo.
