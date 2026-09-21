# HƯỚNG DẪN SỬ DỤNG TOOLRECAP V2 (v0.5.2)

ToolRecap V2 là ứng dụng tự động hóa hoàn toàn quy trình biên tập và sản xuất video recap tiếng Anh chất lượng cao từ các tập phim hoặc cả mùa phim.

---

## 1. Cách mở và khởi chạy ứng dụng

- **Khởi chạy ứng dụng**:
  - Nhấp đúp chuột vào tệp `ToolRecapV2.exe` (hoặc tệp lệnh `Chay-ToolRecapV2.cmd`).
  - Ứng dụng hoạt động theo dạng độc lập (Portable), đã tích hợp sẵn môi trường chạy và công cụ FFmpeg, không yêu cầu cài đặt thêm phần mềm phụ trợ.

---

## 2. Cách sử dụng cơ bản

### 2.1. Chọn nguồn video
- **Tập đơn lẻ (Single Episode)**: Nhấn **📁 Select File** để chọn 1 tệp video duy nhất (.mp4, .mkv, .mov, .avi, .webm, .m4v, .ts).
- **Trọn bộ mùa phim (Season Multi-Source)**: Nhấn **📂 Select Folder** để chọn thư mục chứa các tập phim. Hệ thống quét trực tiếp các tệp video trong thư mục (direct-only) và tự động sắp xếp theo thứ tự tập tự nhiên (ep1, ep2, ep10...).

### 2.2. Cấu hình kịch bản và AI Gateway (Settings)
Nhấn biểu tượng **⚙ Settings** trên thanh công cụ để mở cửa sổ cài đặt gồm 4 thẻ:
1. **Recap**: Thiết lập ngôn ngữ kịch bản, thể loại phim, bản quyền và nhập câu nhắc biên tập tại ô **Recap Prompt**.
2. **AI Gateway**: Cấu hình kết nối AI Gateway (Endpoint, API Key), lựa chọn mô hình Scanner (quét bằng chứng) và Finalizer (biên tập kịch bản).
3. **Voice**: Lựa chọn 12 giọng đọc chuẩn thiết kế, phong cách diễn đọc, kiểm tra nghe thử giọng và các thông số hòa âm chuyên nghiệp (Auto-ducking, Loudness -14 LUFS).
4. **Render and Output**: Lựa chọn chất lượng video, bật/tắt tăng tốc phần cứng GPU (NVENC/AMF/QSV), tùy chọn nhúng phụ đề và thư mục xuất bản.

### 2.3. Bắt đầu sản xuất
- Nhấn nút **Start Creating Recap Videos** để bắt đầu chu trình sản xuất tự động.
- Để dừng quá trình xử lý bất kỳ lúc nào, nhấn nút **⏹ Stop**. Hệ thống sẽ dừng an toàn các tiến trình con mà không để lại tiến trình mồ côi.

---

## 3. Bộ kết xuất đa nguồn mạnh mẽ (Robust Multi-Source Renderer v0.5.2)

Phiên bản v0.5.2 nâng cấp toàn diện bộ kết xuất đa nguồn nhằm đảm bảo ghép nối mượt mà và chuẩn hóa mọi tệp đầu ra:

- **Đường dẫn nhanh ghép nối trực tiếp (Strict signature fast path)**:
  Hệ thống tự động so sánh chữ ký kỹ thuật (StreamSignature) của các phân đoạn bao gồm: codec video/audio, độ phân giải, tốc độ khung hình (fps), định dạng điểm ảnh (pix_fmt), tỉ lệ pixel (SAR), tần số lấy mẫu (sample_rate) và số kênh âm thanh (channels). Khi các phân đoạn hoàn toàn đồng nhất về chữ ký kỹ thuật, hệ thống kích hoạt đường dẫn nhanh ghép nối trực tiếp (stream-copy concat demuxer), giúp hoàn thành ghép nối gần như tức thì mà không cần nén lại, bảo toàn nguyên vẹn chất lượng gốc.

- **Hồ sơ chuẩn hóa video và âm thanh (Normalized video/audio profile)**:
  Khi các đoạn nguồn có thông số kỹ thuật không đồng nhất (khác độ phân giải, lệch tốc độ khung hình, khác số kênh âm thanh), hệ thống tự động chuyển sang chế độ chuẩn hóa hoàn toàn trước khi ghép:
  - Video được chuẩn hóa về kích thước chẵn (even width/height) với tỉ lệ chuẩn (setsar=1), chuyển đổi định dạng yuv420p và đồng bộ tốc độ khung hình.
  - Âm thanh được tự động chuyển đổi sang tần số mẫu 48.000 Hz (48 kHz), định dạng fltp và cấu hình 2 kênh stereo đồng nhất.

- **Xử lý âm thanh im lặng tự động (Silent audio handling)**:
  Đối với các đoạn video nguồn không có luồng âm thanh hoặc phân đoạn áp dụng chính sách tắt tiếng gốc (MUTE_ORIGINAL), hệ thống tự động tạo luồng âm thanh im lặng chuẩn (anullsrc stereo 48 kHz) có độ dài khớp chính xác với độ dài đoạn video. Điều này ngăn ngừa hoàn toàn hiện tượng lệch luồng (stream mismatch) hoặc mất đồng bộ giữa hình ảnh và âm thanh khi ghép nối với các đoạn có âm thanh khác.

- **Xác thực đầu ra nghiêm ngặt (Validation)**:
  Mỗi tệp sau khi cắt đoạn và sau khi ghép nối hoàn chỉnh đều được kiểm tra độc lập qua công cụ thăm dò media (ffprobe):
  - Xác thực thời lượng thực tế của tệp so với thời lượng dự kiến.
  - Xác thực sự hiện diện của luồng video và luồng âm thanh hợp lệ.
  - Đảm bảo đầu ra xuất bản cuối cùng luôn có đủ 3 tệp thành phẩm chuẩn xác: `{tên_video}.mp4`, `{tên_video}.original.srt`, `{tên_video}.narration.srt`.

- **Tiếp tục theo từng đầu ra và phân định lỗi chi tiết (Per-output resume & errors)**:
  - Cơ chế ghi nhớ trạng thái cho phép xử lý độc lập từng video đầu ra. Nếu một dự án có nhiều video đầu ra và bị gián đoạn, khi chạy lại hệ thống sẽ tự động nhận diện chữ ký kết xuất (render signature) của các video đã hoàn thành trước đó để bỏ qua (skip), chỉ tiếp tục kết xuất các video còn lại mà không làm mất dữ liệu đã tạo.
  - Mọi sự cố kỹ thuật đều được phân loại chính xác theo từng giai đoạn thực thi (cắt đoạn - CUT, ghép nối - CONCAT, tạo giọng đọc - SYNTHESIS, hòa âm - MIX, xác thực - VALIDATION) kèm định danh video đầu ra và thông điệp lỗi cụ thể, giúp dễ dàng chẩn đoán và khắc phục.

---

## 4. Dữ liệu lưu trữ ở đâu

- **Thư mục xuất bản video**: Mặc định nằm tại thư mục do người dùng cấu hình trong Settings (hoặc thư mục con `outputs/` trong thư mục dữ liệu). Mỗi video recap hoàn thành được đặt trong một thư mục riêng mang tên video, chứa đúng 3 tệp thành phẩm:
  1. `{tên_video}.mp4`: Video hoàn chỉnh.
  2. `{tên_video}.original.srt`: Phụ đề thoại gốc giữ lại trong video.
  3. `{tên_video}.narration.srt`: Phụ đề giọng đọc thuyết minh AI.
- **Dữ liệu ứng dụng và bộ nhớ đệm (Cache)**: Toàn bộ cấu hình cài đặt, cơ sở dữ liệu hàng đợi dự án, bộ nhớ đệm phụ đề và mô hình giọng đọc được lưu trữ độc lập tại:
  `%LOCALAPPDATA%\ToolRecapV2`
  Đảm bảo dữ liệu người dùng không bị mất khi cập nhật phiên bản mới của ứng dụng.

---

## 5. Xử lý sự cố thường gặp

- **Không tạo được video (0 output)**:
  Kiểm tra lý do hiển thị trên giao diện: nếu là "Genuine Zero", nghĩa là nội dung phim không có tình tiết nào khớp với yêu cầu của Recap Prompt; người dùng có thể điều chỉnh lại câu nhắc để quét nội dung phù hợp. Nếu là lỗi kỹ thuật (kết nối mạng, giải mã), giao diện sẽ hiển thị chi tiết mã lỗi để xử lý.
- **Lỗi kết nối AI Gateway**:
  Kiểm tra địa chỉ Endpoint và API Key trong thẻ Cài đặt -> AI Gateway. Đảm bảo cổng dịch vụ AI nội bộ (mặc định 20128) đang mở và hoạt động bình thường.
- **Lỗi không tìm thấy công cụ FFmpeg**:
  Bản phát hành Portable đã tích hợp sẵn FFmpeg trong thư mục `runtime/ffmpeg/bin`. Không di chuyển hoặc xóa thư mục runtime này khỏi ứng dụng.
