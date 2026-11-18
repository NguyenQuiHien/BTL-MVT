# Dữ liệu bàn giao của thành viên 4

Các file trong kb1, kb2, kb3 được sao chép nguyên trạng từ demo.zip do nhóm cung cấp ngày 17/11/2026. Mỗi thư mục có log 36.000 yêu cầu, bảng 36 lần chạy theo seed, bảng tổng hợp và ảnh PNG.

- Topology được tác giả khai báo: nobel-eu.
- Hai thuật toán: KSP-FF, LS-BM-KSP.
- Tải: 50, 100, 150, 200, 250, 300 Erlang; seed: 42, 43, 44.
- 1.000 yêu cầu mỗi lần chạy, không loại warm-up trong bảng đã gửi.
- KB1: 320 slot, bit rate 10/40/100/400; KB2: 80 slot, cùng bit rate; KB3: 320 slot, bit rate 40/100/400.

Đây là dữ liệu cũ, không phải dữ liệu tạo lại bằng phiên bản tích hợp. Gói gốc chưa có config.json/commit/phần cứng và chưa có kết quả quét công suất. Không suy ra đầy đủ cấu hình vật lý chỉ từ CSV.

`util` và biểu đồ sử dụng phổ cũ được lấy sau env.step, tức có thể sau khi tiến thời gian và giải phóng dịch vụ. Phiên bản mới lấy ngay sau hành động, trước khi tiến thời gian. Không ghép hai cách lấy mẫu thành một chuỗi so sánh. Thời gian mới bao gồm kiểm tra agent và chụp trạng thái nên không so sánh trực tiếp với thời gian cũ.

Mỗi bảng tổng hợp hiển thị tỷ lệ từ 0 đến 1. Các cột std là độ lệch chuẩn giữa seed, không phải khoảng tin cậy. Tên KSP-FF ánh xạ tới jocn_ksp_ff_bm, LS-BM-KSP ánh xạ tới jocn_ls_bm_ksp của ONG.
