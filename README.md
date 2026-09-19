# ToolRecap V2 — Ứng Dụng Tự Động Hóa Sản Xuất Video Recap (Portable Windows)

ToolRecap V2 là ứng dụng máy tính dành riêng cho Windows giúp tự động hóa 100% quy trình sản xuất video recap (tóm tắt phim/truyền hình) tiếng Anh chỉ với một cú nhấp chuột: quét tập phim, phân tích nội dung thoại thực tế (qua phụ đề companion SRT hoặc nhận diện giọng nói faster-whisper / API), đọc lời dẫn chân thực bằng AI (Piper TTS), cắt cảnh theo nhịp kịch bản, nhúng phụ đề và xuất bản video hoàn chỉnh với tăng tốc phần cứng Hybrid GPU.

---

## 1. Cách Mở Ứng Dụng

Ứng dụng chạy dưới dạng **bản Portable độc lập** (không cần cài đặt Python, không cần cài đặt FFmpeg hay cấu hình môi trường máy tính phức tạp):

- **Cách 1 (Khuyên dùng)**: Nhấp đúp chuột vào tệp `release\ToolRecapV2\ToolRecapV2.exe` hoặc tệp `Chay-ToolRecapV2.cmd`.
- **Cách 2 (Môi trường phát triển)**: Chạy lệnh `python auto_main.py` từ thư mục dự án.

---

## 2. Hướng Dẫn Sử Dụng (4 Bước Đơn Giản)

1. **Chọn nguồn video**:
   - Nhấn nút **"📁 Chọn 1 file video..."** để xử lý 1 tập lẻ.
   - Hoặc nhấn **"📂 Chọn thư mục chứa video..."** để xử lý cả mùa. Ứng dụng quét trực tiếp (không quét đệ quy các thư mục con), chỉ lấy các định dạng video được hỗ trợ (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`, `.ts`), và tự động sắp xếp theo thứ tự tập tự nhiên (`ep1, ep2, ep10`).
2. **Chọn giọng đọc thuyết minh**:
   - Ở khung bên phải, chọn giọng đọc tiếng Anh mong muốn từ danh sách (Lessac, Ryan, Alba, Alan).
   - Nhấn **"🔊 Nghe thử giọng"** để nghe âm thanh mẫu. Mô hình giọng đọc sẽ được tải tự động (lazy download) ở lần nghe đầu tiên và lưu vào bộ nhớ đệm.
3. **Bắt đầu sản xuất tự động**:
   - Nhấn nút lớn màu xanh **"▶ Bắt đầu tự động"**.
   - Ứng dụng chạy tuần tự từng video qua các giai đoạn:
     - **Phân tích nội dung thoại thực tế**: Đọc phụ đề đi kèm (`.srt`) hoặc dùng mô hình `faster-whisper` nội bộ / API OpenAI Whisper để trích xuất hội thoại thực tế (không dùng kịch bản ngẫu nhiên hay generic placeholder).
     - **Tổng hợp giọng đọc AI (TTS)**: Đọc lời dẫn thuyết minh bằng Piper TTS thật (không dùng âm mẫu giả).
     - **Cắt cảnh & Ghép âm thanh**: Cắt các phân đoạn phim tương ứng thời lượng kịch bản, ghép giọng dẫn và tạo phụ đề SRT.
     - **Render video**: Mã hóa video đầu ra bằng quy trình tăng tốc Hybrid GPU hoặc CPU.
4. **Xem kết quả**:
   - Nhấn nút **"📂 Mở thư mục kết quả"** để mở thư mục chứa video recap và tệp phụ đề đã hoàn thành.

---

## 3. Tăng Tốc Phần Cứng Hybrid GPU & Bounded Fallback

ToolRecap V2 hỗ trợ cơ chế tăng tốc phần cứng thông minh:
- **Quy trình Hybrid**: Tự động kết hợp giải mã phần cứng Intel QSV (hoặc CPU) và mã hóa phần cứng rời NVIDIA NVENC (hoặc AMD AMF / Intel QSV) để tối ưu hóa hiệu suất và chất lượng hình ảnh.
- **Dự phòng có giới hạn (Bounded Fallback)**: Nếu giải mã hoặc mã hóa phần cứng gặp sự cố (thiếu driver, định dạng không hỗ trợ), ứng dụng tự động chuyển đổi sang phương án dự phòng tiếp theo và cuối cùng là mã hóa phần mềm CPU `libx264`, đảm bảo tiến trình render không bao giờ bị gián đoạn hay treo cứng.

---

## 4. Dừng An Toàn (Safe Cancellation)

- Trong khi render, nút **"⏹ Dừng xử lý"** sẽ sáng lên.
- Khi nhấn dừng:
  - Ứng dụng gửi tín hiệu dừng ngay lập tức.
  - Tự động đóng cây tiến trình con FFmpeg (`taskkill /F /T /PID`).
  - Mở khóa lại toàn bộ nút bấm trên giao diện (`▶ Bắt đầu tự động`, chọn file/thư mục).
  - Cập nhật trạng thái các tập chưa chạy thành `CANCELLED`.

---

## 5. Nơi Lưu Trữ Dữ Liệu

Tất cả dữ liệu người dùng được lưu trữ hoàn toàn tách biệt ngoài thư mục ứng dụng tại:
```
%LOCALAPPDATA%\ToolRecapV2
```
Bao gồm:
- `settings.json`: Cấu hình chất lượng video, GPU, nhúng phụ đề, cấu hình STT. Lưu ý: API Key được lưu dưới dạng văn bản thuần (plain text) trong tệp cấu hình cục bộ theo người dùng (%LOCALAPPDATA%), không được mã hóa mật mã (encryption). Không chia sẻ tệp này hoặc dùng khóa có quyền hạn cao trên máy dùng chung.
- `projects.json`: Lịch sử và trạng thái hàng đợi các tập phim (lưu trữ nguyên tử, tự động phục hồi nếu mất điện/đóng đột ngột).
- `models\voices\`: Bộ nhớ đệm các mô hình giọng đọc Piper ONNX tải về (bảo tồn nguyên vẹn khi nâng cấp ứng dụng).
- `models\stt\`: Bộ nhớ đệm mô hình nhận diện giọng nói faster-whisper (tải tự động lần đầu, tái sử dụng không cần tải lại).
- `logs\`: Nhật ký hoạt động và thông tin lỗi.

---

## 6. Cập Nhật Ứng Dụng & Phân Biệt VoiceStudio Subsystem

- **Cập nhật ToolRecap V2**: Tự động kiểm tra GitHub Releases chính thức từ `longthao9820-alt/tool-recap-v2`. Hộp thoại hiển thị rõ ràng cả phiên bản hiện tại và phiên bản mới nhất. Bản cập nhật bắt buộc phải có mã băm SHA256 hợp lệ, áp dụng qua kịch bản hoán đổi có rollback và bảo toàn dữ liệu người dùng.
- **Phân biệt VoiceStudio Subsystem**:
  - Bản phát hành chính thức `debpalash/VoiceStudio v0.5.3` đã được kiểm tra trực tiếp: hiện tại tác giả chỉ cung cấp bản dựng desktop độc lập (AGPL), chưa có gói adapter V2 tương thích.
  - Hiện tại, hệ thống giọng đọc **Piper TTS hoạt động đầy đủ, sẵn sàng sử dụng ngay**.
  - Cơ chế cập nhật VoiceStudio trong ToolRecap V2 đã sẵn sàng: khi có gói adapter tương thích chính thức với chữ ký / mã băm SHA256 (hoặc GitHub digest) hợp lệ và `voice_manifest.json` chuẩn, hệ thống sẽ tự động xác thực và cài đặt an toàn. Tuyệt đối không tự ý sao chép ứng dụng desktop hoặc giả định adapter đã được cài đặt khi chưa có gói hợp lệ.

---

## 7. Giới Hạn Của Hệ Thống

- **Chất lượng kịch bản thuyết minh (Extractive Recap)**: Kịch bản thuyết minh được trích xuất và tổng hợp dựa trên lời thoại thực tế từ phụ đề SRT đi kèm hoặc nhận diện âm thanh (faster-whisper / API), đảm bảo bám sát nội dung. Tuy nhiên, đây là phương pháp tóm tắt trích đoạn (extractive recap) dựa trên phân đoạn và năng lượng thoại thực tế, không phải là mô hình LLM suy luận phức tạp.
- **Phân tích nội dung thoại**: Nếu video không có tiếng nói và không có tệp phụ đề SRT đi kèm, ứng dụng sẽ phân đoạn cảnh theo dòng thời gian video tự nhiên.
- **Tải mô hình ban đầu**: Lần đầu tiên sử dụng một giọng đọc hoặc mô hình STT nội bộ, ứng dụng cần kết nối Internet để tải mô hình về máy (tiến trình tải được hiển thị trực tiếp trên giao diện).
- **Tăng tốc GPU**: Yêu cầu máy tính có card đồ họa tương thích và đã cài đặt driver chính thức (NVIDIA, AMD hoặc Intel).

---

## 8. Xử Lý Các Lỗi Thường Gặp

| Hiện tượng | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **Không tìm thấy video trong thư mục** | Video nằm trong các thư mục con sâu hơn. | Đưa video ra thư mục chính, vì ứng dụng quét trực tiếp (non-recursive) để đảm bảo chính xác thứ tự tập. |
| **Render chậm hoặc CPU cao** | Máy tính chưa cài driver GPU phù hợp. | Vào **⚙ Cài đặt**, kiểm tra mục GPU. Nếu không có card rời, ứng dụng tự động dùng CPU `libx264` rất ổn định. |
| **Lỗi dung lượng ổ đĩa** | Ổ đĩa chứa thư mục xuất video bị đầy. | Nhấn **⚙ Cài đặt** và đổi thư mục xuất sang ổ đĩa còn nhiều dung lượng trống. |
| **Lỗi khởi động ứng dụng** | Thiếu file thư viện hoặc FFmpeg. | Chạy lệnh `python auto_main.py --self-check` để xem báo cáo kiểm tra chi tiết. |

---

## 9. Đóng Gói Bản Phát Hành (Release)

Để đóng gói ra bản Portable hoàn chỉnh:
1. Nhấp đúp vào tệp `Dong-Goi-ToolRecapV2.cmd` (hoặc chạy lệnh `python build_exe.py`).
2. Script sẽ tự động:
   - Dựng bản portable mới vào `release\ToolRecapV2\`.
   - Sao chép `ffmpeg.exe`, `ffprobe.exe`, `LICENSE`, `THIRD_PARTY_LICENSES.md`, và tệp hướng dẫn sử dụng.
   - Kiểm tra khả năng mã hóa NVENC/AMF/QSV của binary FFmpeg.
   - Chạy kiểm tra tự động `--version` và `--self-check` trên tệp `.exe`.
   - Tạo gói nén `release\ToolRecapV2-v0.1.0-windows-portable.zip` và tệp mã băm companion `ToolRecapV2-v0.1.0-windows-portable.zip.sha256.txt`.

---

## 10. Giấy Phép & Nguồn Gốc Thành Phần

- **ToolRecap V2 Core**: Bản quyền mã nguồn mở MIT.
- **FFmpeg / FFprobe**: Bản dựng LGPL / GPL từ gyan.dev.
- **Piper TTS**: Bản quyền mã nguồn mở MIT / Apache-2.0 từ Rhasspy.
- **faster-whisper**: Bản quyền mã nguồn mở MIT từ Systran.
- **Biểu tượng (Icon)**: Thiết kế vector tạo bằng Pillow, thuộc sở hữu dự án.
