# ToolRecap V2 — Ứng Dụng Tự Động Hóa Sản Xuất Video Recap (Portable Windows v0.3.1)

ToolRecap V2 là ứng dụng máy tính dành riêng cho hệ điều hành Windows giúp tự động hóa 100% quy trình sản xuất video recap (tóm tắt phim, truyền hình, tài liệu) tiếng Anh chất lượng cao chỉ với một cú nhấp chuột: quét nguồn video, phân tích và trích xuất hội thoại thực tế (qua phụ đề companion SRT/VTT/ASS, phụ đề đồ họa bitmap PGS/VobSub qua RapidOCR/AI Vision, hoặc nhận diện giọng nói STT faster-whisper), suy luận kịch bản phân đoạn 2 giai đoạn (Scanner -> Finalizer) qua AI Gateway chuẩn OpenAI, tổng hợp giọng dẫn thuyết minh chân thực với 12 giọng thiết kế chuẩn, phối trộn âm thanh tự động (Real Audio Mix with Auto-Ducking), nhúng phụ đề và xuất bản video hoàn chỉnh với tăng tốc phần cứng Hybrid GPU.

---

## 1. Cách Mở Ứng Dụng

Ứng dụng chạy dưới dạng **bản Portable độc lập** (không cần cài đặt Python, không cần cài đặt FFmpeg hay thiết lập biến môi trường hệ thống):

- **Cách 1 (Khuyên dùng)**: Nhấp đúp chuột vào tệp `ToolRecapV2.exe` (hoặc tệp `Chay-ToolRecapV2.cmd`) trong thư mục bản dựng.
- **Cách 2 (Môi trường phát triển)**: Chạy lệnh `python auto_main.py` từ thư mục gốc của dự án.

---

## 2. Quy Trình Sử Dụng: Single vs Season

Ứng dụng hỗ trợ hai phương thức chọn nguồn video:

1. **Xử lý tập đơn lẻ (Single Episode)**:
   - Nhấn nút **"📁 Select File"** để chọn 1 tệp video duy nhất.
   - Phù hợp khi bạn muốn recap nhanh một tập phim hoặc video ngắn độc lập.
2. **Xử lý trọn bộ mùa phim (Season Multi-Episode)**:
   - Nhấn nút **"📂 Select Folder"** để chọn thư mục chứa tất cả các tập của mùa phim.
   - Hệ thống sẽ kích hoạt quy trình phân tích mùa (Season Analysis), liên kết sự kiện giữa các tập và khai phá các ứng viên kịch bản xuyên suốt mùa (Season Arc, Storylines).
3. **Quét trực tiếp không đệ quy (Source Direct-Only)**:
   - Ứng dụng quét trực tiếp các tệp nằm ngay trong thư mục được chọn, **tuyệt đối không quét đệ quy vào các thư mục con**.
   - Chỉ lọc và nhận diện các định dạng video được hỗ trợ: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`, `.ts`.
   - Tự động sắp xếp video theo thứ tự tập tự nhiên (natural sort: `ep1, ep2, ep10`), tránh sai lệch thứ tự tập phim.
4. **Bắt đầu sản xuất**:
   - Nhấn nút lớn màu xanh **"▶ Start Creating Recap Videos"**.
   - Theo dõi tiến trình thời gian thực trên bảng hàng đợi (Episode, Source Video, Stage, Progress, Status).
5. **Xem sản phẩm hoàn thành**:
   - Nhấn nút **"📂 Open Output Folder"** để mở thư mục kết quả.

---

## 3. Các Giai Đoạn Xử Lý (Phases)

Mỗi tập phim và mùa phim được xử lý tuần tự qua các giai đoạn nghiêm ngặt:

1. **`media_probe`**: Thăm dò và phân tích thông số kỹ thuật video (độ phân giải, thời lượng, luồng hình ảnh, luồng âm thanh) bằng FFprobe nhúng.
2. **`subtitles`**: Khám phá và trích xuất phụ đề (ưu tiên tiếng Anh, xử lý tệp phụ đề ngoài hoặc luồng phụ đề nhúng trong container).
3. **`scanner`**: Quét từng phân đoạn transcript/video qua Scanner AI để trích xuất bằng chứng câu chuyện (narrative evidence) gắn mốc thời gian tuyệt đối.
4. **`season_barrier`** (Chế độ Season): Điểm đồng bộ hóa bắt buộc — toàn bộ các tập trong mùa phải hoàn thành giai đoạn Scanner trước khi bước vào kết nối mùa.
5. **`season_connecting`** (Chế độ Season): Phân tích mạng lưới liên kết sự kiện, xung đột và nhân vật xuyên suốt toàn bộ các tập của mùa.
6. **`season_mining`** (Chế độ Season): Khai phá và tuyển chọn các ứng viên kịch bản hoàn chỉnh (Candidate Mining).
7. **`output_plan_ready`**: Hoàn thiện kế hoạch phân đoạn và kịch bản recap.
8. **Giai đoạn Kết xuất (Rendering)**:
   - Tổng hợp lời dẫn thuyết minh (Timeline narration) bằng mô hình giọng đọc AI.
   - Phối trộn âm thanh thông minh (Real Audio Mix).
   - Tạo tệp phụ đề SRT đồng bộ chính xác đến từng mili-giây.
   - Mã hóa video đầu ra hoàn chỉnh bằng bộ tăng tốc Hybrid GPU hoặc CPU.

---

## 4. Cấu Hình Cài Đặt (Chính Xác 4 Tab)

Nhấn nút **"⚙ Settings"** trên thanh công cụ chính để mở cửa sổ cấu hình trung tâm với đúng 4 tab:

### Tab 1: Recap (Kịch bản & Phim)
- **Recap language**: Ngôn ngữ kịch bản (`en-US`, `en-GB`).
- **Recap mode**: Chế độ recap (`MAIN_STORIES` — tập trung tuyến truyện chính, hoặc `FULL_EPISODE` — tóm tắt toàn diện tập).
- **Content type**: Phân loại nội dung (`US_TV_SHOW`, `DE_GERMAN_SOAP`, `BODYCAM`, `FEATURE_FILM`, `OTHER`).
- **Footage rights**: Trạng thái bản quyền tư liệu (`UNVERIFIED`, `OWNED`, `LICENSED`, `FIRST_PUBLICATION_RIGHTS`, `FAIR_USE`).
- **Recap Prompt**: Khung soạn thảo prompt chỉ dẫn cho AI tạo kịch bản, kèm nút **"Reload Default Prompt"** để khôi phục chỉ dẫn chuẩn của từng thể loại phim.

### Tab 2: AI Gateway (Kết Nối Mô Hình AI)
- **Kích hoạt AI Gateway**: Bật/tắt phân tích kịch bản qua API OpenAI-compatible (khi tắt, hệ thống sử dụng thuật toán tóm tắt trích xuất offline nội bộ).
- **API endpoint**: Địa chỉ cổng Gateway (mặc định: `http://127.0.0.1:20128/v1`).
- **API key**: Khóa truy cập API (được lưu an toàn cục bộ trong `settings.json`, có nút Hiện/Ẩn, không bao giờ lộ ra nhật ký hoạt động).
- **Scanner model**: Mô hình quét bằng chứng (mặc định: `sub`, thinking: `max`).
- **Scanner model supports image/Vision input**: Checkbox rõ ràng cho phép gửi hình ảnh phụ đề/khung cảnh lên mô hình Vision khi cần thiết.
- **Finalizer model**: Mô hình tổng hợp kịch bản (mặc định: `prime`, thinking: `high`).
- **Song song (parallelism)**: Số luồng phân tích Scanner đồng thời (1 đến 4 luồng, mặc định 2).
- **Độ dài đoạn (chunk seconds)**: Thời lượng mỗi phân đoạn transcript (60s đến 900s, mặc định 300s).
- **Nút Test**: Kiểm tra kết nối độc lập cho Scanner và Finalizer ngay trong bảng cài đặt.

### Tab 3: Voice (Giọng Đọc & Phối Trộn Âm Thanh)
- **Language & Voice selection**: Lựa chọn từ 12 giọng đọc thiết kế chuẩn tiếng Anh (Neighbor, Companion, v.v., phân loại theo vùng `en-US` và `en-GB`, nam/nữ).
- **Voice style**: Chọn phong cách biểu cảm (`film_recap`, `storytelling`, `documentary`, `crime_thriller`, `drama`, `soap_emotional`, `energetic`, `neutral`).
- **🔊 Nghe thử giọng**: Nghe trước giọng đọc mẫu kèm thanh hiển thị tiến trình tải mô hình và nút dừng nghe tức thì.
- **🎙 Cập nhật VoiceStudio**: Mở hộp thoại kiểm tra và cấu hình môi trường giọng đọc mở rộng.
- **Phối trộn âm thanh (Audio Mix)**:
  - *Original audio*: Mức tăng/giảm âm lượng âm thanh phim gốc (-60.0 dB đến +24.0 dB).
  - *Commentary voice*: Mức tăng/giảm âm lượng giọng thuyết minh (-60.0 dB đến +24.0 dB).
  - *Auto-duck original audio during commentary*: Tự động hạ âm lượng phim gốc khi có giọng thuyết minh cất lên.
  - *Ducking amount*: Mức độ giảm âm nền khi ducking (mặc định -12.0 dB).
  - *Target loudness*: Chuẩn hóa độ ồn tổng thể theo chuẩn phát thanh (mặc định -14.0 LUFS).
  - *True peak*: Mức trần âm thanh tối đa chống vỡ tiếng (mặc định -1.0 dBTP).

### Tab 4: Render and Output (Xuất Bản & Hệ Thống)
- **Chất lượng video**: `standard` (tiết kiệm dung lượng), `high` (chất lượng cao), `source` (giữ nguyên độ phân giải nguồn).
- **Bật tăng tốc phần cứng GPU**: Tận dụng NVIDIA NVENC, AMD AMF hoặc Intel QSV để mã hóa video siêu tốc.
- **Nhúng thẳng phụ đề vào video (Burn subtitles)**: Bật để ghi trực tiếp chữ phụ đề lên hình ảnh video đầu ra.
- **Thư mục xuất video**: Chọn vị trí lưu trữ thành phẩm recap.
- **Hệ thống phụ & Cập nhật**: Kiểm tra phiên bản và môi trường phụ trợ VoiceStudio.

*(Lưu ý: Hệ thống không để lộ tab STT riêng biệt; tính năng nhận diện giọng nói STT hoạt động tự động ngầm bên trong. Các thông số độ tin cậy AI như phase timeouts, số lượt thử lại 3 attempts, kích thước batching 3-4 cũng được quản lý hoàn toàn nội bộ, không đưa vào giao diện Settings nhằm tránh làm phức tạp trải nghiệm người dùng).*

---

## 5. Độ Tin Cậy AI Gateway (AI Reliability & Resilience)

Quy trình phân tích kịch bản bằng AI Gateway được thiết kế với cơ chế độ tin cậy công nghiệp:

1. **Phase Timeouts nội bộ**: Mỗi giai đoạn phân tích AI (`scanner`, `season_connecting`, `season_mining`, `finalizer`) được ấn định thời hạn chờ (timeout) nội bộ riêng biệt, tối ưu theo khối lượng tính toán. Người dùng không phải bận tâm cấu hình thủ công.
2. **Tự động thử tối đa 3 lần**: Khi gặp lỗi timeout, mất kết nối, HTTP 429 hoặc lỗi máy chủ tạm thời, hệ thống tự động thử lại với khoảng chờ 5 và 15 giây. Nút Stop ngắt khoảng chờ ngay lập tức.
3. **Phân tích mùa theo lô 3–4 tập**: Mỗi request Season Connection chỉ nhận compact summaries của tối đa 3–4 tập. Các kết quả lô được hợp nhất phân cấp để kết nối cốt truyện xuyên suốt mùa.
4. **Tóm tắt cô đọng, Cache & Khôi phục (Compact Summaries, Caching & Resume)**:
   - Tự động sinh tóm tắt cô đọng cho các đoạn hội thoại dài, giảm thiểu token dư thừa.
   - Lưu trữ kết quả phân tích theo mã băm định danh nội dung (content hash cache).
   - Hỗ trợ khôi phục và tiếp tục (resume) xử lý ngay tại giai đoạn dang dở mà không cần quét lại từ đầu.
5. **Phân định rõ quyền sở hữu tiến trình & lỗi (Progress & Error Ownership)**:
   - Trạng thái tiến trình được gán rõ ràng theo từng giai đoạn và từng tập phim.
   - Khi phát sinh lỗi, hệ thống phân định chính xác phạm vi (tập phim đơn lẻ hay toàn mùa) và thông báo lỗi rõ ràng trên hàng đợi, không làm gián đoạn hay làm sai lệch dữ liệu của các tập thành công khác.
6. **Cài đặt tinh gọn**: Toàn bộ các cơ chế độ tin cậy trên hoạt động ngầm (internal), không đưa các thiết lập kỹ thuật này vào bảng cài đặt.

---

## 6. Trích Xuất Hội Thoại & Phụ Đề: PGS / VobSub / RapidOCR / AI Vision / STT

Ứng dụng sở hữu cơ chế bóc tách hội thoại đa tầng hiện đại:

1. **Ưu tiên phụ đề tiếng Anh**: Tự động dò tìm luồng phụ đề tiếng Anh trong tệp đa phương tiện hoặc các tệp phụ đề sidecar cùng tên (`.srt`, `.vtt`, `.ass`).
2. **Phụ đề bitmap PGS / VobSub**: Đối với các nguồn phim Blu-ray (PGS `.sup`) hoặc DVD (VobSub `.sub`/`.idx`):
   - Hệ thống giải mã đồ họa trực tiếp và nhận dạng chữ qua thư viện **RapidOCR ONNX** nội bộ mà không cần cài đặt thêm phần mềm ngoài.
   - **AI Vision Fallback**: CHỈ KHI người dùng đánh dấu chọn checkbox *"Scanner model supports image/Vision input"* trong cài đặt AI Gateway, các dòng phụ đề mờ khó đọc mới được gửi lên mô hình Vision để hỗ trợ giải mã.
3. **STT Dự phòng nội bộ (Internal STT)**:
   - Nếu video không có phụ đề hoặc quá trình OCR không tìm thấy nội dung thoại, hệ thống tự động kích hoạt bộ nhận diện giọng nói **faster-whisper** nội bộ (chạy tối ưu hóa trên CPU) hoặc API Whisper để bóc băng trực tiếp từ âm thanh của phim.

---

## 7. 12 Giọng Thiết Kế Chuẩn & Cơ Chế Tải Runtime/Model Lần Đầu

- **Danh mục 12 giọng chính thức**: Kế thừa kiến trúc VoiceStudio / OmniVoice với 12 nhân vật giọng đọc chuyên biệt (6 giọng `en-US` và 6 giọng `en-GB`, nam/nữ, phù hợp cho tóm tắt phim, kịch tính, tài liệu).
- **Tải lần đầu (First-use download)**:
  - Ở lần đầu sử dụng một giọng đọc hoặc tính năng nâng cao, ứng dụng sẽ tải môi trường Python độc lập và trọng số mô hình OmniVoice (dung lượng lớn khoảng vài GB).
  - Tiến trình tải được hiển thị rõ ràng từng phần trăm và dung lượng byte trên giao diện, hỗ trợ hủy an toàn.
  - Toàn bộ runtime và mô hình sau khi tải được lưu vào thư mục bộ nhớ đệm tại `%LOCALAPPDATA%\ToolRecapV2\voice_subsystem` và cache mô hình của máy. Các lần chạy tiếp theo sẽ hoạt động tức thì, hoàn toàn offline.
- **Tương thích nội bộ (Built-in Fallback)**: Ứng dụng luôn tích hợp sẵn runtime giọng nói Piper TTS nội bộ, sẵn sàng hoạt động ngay cả khi chưa tải gói OmniVoice lớn.

---

## 8. Sản Phẩm Xuất Bản: Đúng 3 Tệp Thành Phẩm (Outputs Exact Three)

Mỗi phân đoạn video recap được xuất bản vào một thư mục riêng biệt sạch sẽ, chỉ chứa **ĐÚNG BA TỆP THÀNH PHẨM**:

1. **`{safe_title}.mp4`**: Video recap hoàn chỉnh (hình ảnh khớp nhịp kịch bản, âm thanh hòa trộn giọng dẫn và âm nền, phụ đề nhúng tùy chọn).
2. **`{safe_title}.original.srt`**: Tệp phụ đề các câu thoại gốc của phim được trích dẫn trong bản recap.
3. **`{safe_title}.narration.srt`**: Tệp phụ đề toàn bộ lời dẫn thuyết minh của AI với mốc thời gian chuẩn xác.

Thư mục xuất bản được tự động dọn dẹp sạch sẽ, không chứa bất kỳ tệp tạm, tệp nhật ký hay thư mục con nào.

---

## 9. Dừng An Toàn (Stop / Safe Cancellation)

- Trong quá trình phân tích hoặc kết xuất, nút **"⏹ Stop"** luôn sẵn sàng.
- Khi nhấn nút dừng:
  - Ứng dụng lập tức phát cờ hủy tiến trình an toàn (`cancel_event`).
  - Ngắt quãng tức thì chu kỳ chờ thử lại (backoff delay), hủy bỏ ngay các giai đoạn xử lý kế tiếp (next phases).
  - Đóng sạch sẽ cây tiến trình con FFmpeg bằng lệnh hệ thống (`taskkill /F /T /PID`).
  - Xóa bỏ các tệp tạm thời chưa hoàn thiện.
  - Mở khóa lại toàn bộ các nút bấm trên giao diện và đánh dấu trạng thái của các tập chưa hoàn thành là `CANCELLED`.
  - Tuyệt đối không để xảy ra hiện tượng treo tiến trình nền (orphan process).
  - **Giới hạn kỹ thuật chính xác**: Một yêu cầu HTTP `urlopen` đang gửi nhận dở dang (active in-flight) trên socket chỉ có thể trả về khi nhận phản hồi từ server hoặc khi hết thời gian chờ socket timeout; ngay khi socket hoàn tất hoặc chạm timeout, thao tác Stop lập tức chặn đứng mọi hành động tiếp theo.

---

## 10. Cập Nhật & Lưu Trữ Dữ Liệu

- **Cập nhật ứng dụng ToolRecap V2**: Tự động kiểm tra GitHub Releases chính thức từ `longthao9820-alt/tool-recap-v2`. Bản cập nhật được xác thực mã băm SHA256 trước khi hoán đổi an toàn có cơ chế khôi phục (rollback).
- **Cập nhật VoiceStudio Subsystem**: Kiểm tra và áp dụng gói cập nhật adapter tách biệt, xác thực mã băm SHA256 và manifest hợp lệ.
- **Nơi lưu trữ dữ liệu người dùng**: Toàn bộ cài đặt, lịch sử hàng đợi và bộ nhớ đệm được lưu tại:
  ```
  %LOCALAPPDATA%\ToolRecapV2
  ```
  - `settings.json`: Cấu hình ứng dụng và API Key (lưu văn bản thuần cục bộ).
  - `projects.json`: Lịch sử và hàng đợi dự án (lưu nguyên tử, chống hỏng hóc khi mất điện).
  - `cache\gateway_analysis\`: Bộ nhớ đệm phân tích AI Gateway theo mã băm video, giúp chạy lại không tốn API call.
  - `cache\subtitles\`: Bộ nhớ đệm phụ đề và kết quả OCR.
  - `voice_subsystem\`: Môi trường runtime Python độc lập cho giọng đọc nâng cao.
  - `models\`: Bộ nhớ đệm trọng số mô hình AI (OCR, STT, TTS).
  - `logs\`: Nhật ký hoạt động và thông báo lỗi.

---

## 11. Giới Hạn Của Hệ Thống (Limitations)

- **Mạng Internet lần đầu**: Cần kết nối Internet ổn định ở lần sử dụng đầu tiên để tải các gói runtime và mô hình AI nặng.
- **Yêu cầu phần cứng**: Để đạt tốc độ mã hóa video cao nhất, khuyến nghị máy tính có card đồ họa hỗ trợ NVIDIA NVENC, AMD AMF hoặc Intel QSV. Nếu không có card rời, ứng dụng tự động dùng CPU với bộ mã hóa `libx264` chất lượng cao nhưng thời gian kết xuất sẽ lâu hơn.
- **Độ phụ thuộc vào nguồn thoại**: AI Gateway suy luận và tạo kịch bản dựa trên phụ đề và lời thoại nhận dạng được. Đối với video không có bất kỳ lời thoại hay phụ đề nào, kịch bản recap sẽ chỉ dựa trên thông tin dòng thời gian tổng quát.
- **Giới hạn kết nối mạng HTTP In-Flight**: Yêu cầu mạng HTTP `urlopen` đang gửi nhận trực tiếp trên socket không thể bị ngắt giữa chừng từ bên ngoài Python socket mà phải chờ máy chủ phản hồi hoặc chạm thời gian chờ socket timeout. Nút Stop sẽ ngắt chu kỳ chờ thử lại (backoff) và ngăn không cho các giai đoạn kế tiếp được kích hoạt.

---

## 12. Đóng Gói Bản Phát Hành Portable

Để tạo bản phân phối Portable độc lập:
1. Nhấp đúp vào `Dong-Goi-ToolRecapV2.cmd` (hoặc chạy lệnh `python build_exe.py`).
2. Kịch bản sẽ tự động:
   - Dựng ứng dụng bằng PyInstaller với tệp cấu hình `ToolRecapV2.spec`.
   - Nhúng FFmpeg, FFprobe, giấy phép và tệp hướng dẫn sử dụng vào `release\ToolRecapV2\`.
   - Chạy kiểm tra tự động `--version` và `--self-check` trên tệp thực thi đã dựng.
   - Nén toàn bộ thành `release\ToolRecapV2-v0.3.1-windows-portable.zip` và tạo tệp mã băm companion `ToolRecapV2-v0.3.1-windows-portable.zip.sha256.txt`.

---

## 13. Giấy Phép Bản Quyền (Licenses)

- Mã nguồn chính của ToolRecap V2 được phát hành theo giấy phép **MIT License**.
- Các thành phần bên thứ ba (FFmpeg, Piper, RapidOCR, OmniVoice, faster-whisper, PyTorch, OpenCV, Shapely) tuân theo giấy phép mã nguồn mở tương ứng. Xem chi tiết tại tệp `THIRD_PARTY_LICENSES.md`.
- Người dùng tự chịu trách nhiệm về bản quyền của video nguồn và việc sử dụng các mô hình AI theo điều khoản của nhà cung cấp.
