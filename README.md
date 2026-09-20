# ToolRecap V2 — Ứng Dụng Tự Động Hóa Sản Xuất Video Recap (Portable Windows v0.4.0)

ToolRecap V2 là ứng dụng máy tính dành riêng cho hệ điều hành Windows giúp tự động hóa 100% quy trình sản xuất video recap (tóm tắt phim, truyền hình, tài liệu) tiếng Anh chất lượng cao chỉ với một cú nhấp chuột: quét nguồn video, phân tích và trích xuất hội thoại thực tế (qua phụ đề companion SRT/VTT/ASS, phụ đề đồ họa bitmap PGS/VobSub qua RapidOCR/AI Vision, hoặc nhận diện giọng nói STT faster-whisper), suy luận kịch bản phân đoạn đa tầng có đối soát độ phủ (Scanner -> Coverage Second Pass -> Connection -> Candidate Discovery & Consolidation -> Finalizer) qua AI Gateway chuẩn OpenAI, tổng hợp giọng dẫn thuyết minh chân thực với 12 giọng thiết kế chuẩn, phối trộn âm thanh tự động (Real Audio Mix with Auto-Ducking), nhúng phụ đề và xuất bản video hoàn chỉnh với tăng tốc phần cứng Hybrid GPU.

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

## 3. Các Điểm Cải Tiến Cốt Lõi Trong Phiên Bản v0.4.0

### 3.1. Một Câu Nhắc Điều Khiển Toàn Bộ Chu Trình (One Prompt Controls Full Pipeline)
- Người dùng chỉ cần nhập một câu nhắc duy nhất tại khung **"Recap Prompt"** trong Cài đặt (Tab 1 - Recap).
- Hệ thống tự động phân tích và giải mã câu nhắc này thành các chỉ thị biên tập chuyên biệt cho từng giai đoạn độc lập:
  - *Chỉ thị quét Scanner*: Nhận diện các sự kiện, lời thoại và mâu thuẫn trọng tâm.
  - *Chỉ thị độ phủ (Coverage)*: Xác định các khoảng thời gian hoặc nhân vật cần kiểm tra bổ sung.
  - *Chỉ thị liên kết mùa (Connection)*: Hướng dẫn kết nối các tuyến truyện xuyên suốt các tập.
  - *Chỉ thị ứng viên (Candidate Discovery & Consolidation)*: Định hướng lựa chọn và gom nhóm ý tưởng kịch bản.
  - *Chỉ thị hoàn thiện (Finalizer)*: Quy định giọng văn và cấu trúc phân đoạn kịch bản cuối cùng.
- Không cần cấu hình phức tạp ở nhiều nơi; một chỉ dẫn thống nhất điều phối toàn diện chất lượng nội dung.

### 3.2. Quét Phủ Lần Hai (Coverage Second Pass)
- Hệ thống tự động xây dựng sổ cái theo dõi độ phủ thời gian thực (`Coverage Ledger`) cho từng tập phim, đo lường tỷ lệ bao phủ của lời thoại và các bằng chứng câu chuyện.
- Khi phát hiện khoảng trống thời gian (`gap`) đáng kể hoặc thiếu hụt bằng chứng cho các nhân vật/danh mục chính, hệ thống tự động kích hoạt **lượt quét thứ hai có mục tiêu** (second-pass) nhắm chính xác vào khoảng thời gian đó.
- Đảm bảo trích xuất đầy đủ các diễn biến quan trọng mà không bị sót dữ liệu trong các tập phim dài.

### 3.3. Khám Phá & Hợp Nhất Ứng Viên (Candidate Discovery & Consolidation)
- **Khám phá ứng viên (Discovery)**: Tự động tìm kiếm các ý tưởng kịch bản tiềm năng ở nhiều cấp độ (cảnh đơn lẻ, chuỗi cảnh, hoặc tuyến truyện xuyên suốt cả mùa), gắn chặt với mốc thời gian thực tế.
- **Hợp nhất ứng viên (Consolidation)**: Tự động gom nhóm các ứng viên trùng lặp hoặc bổ trợ cho nhau theo cấu trúc phân cấp thông minh, giữ lại các góc nhìn độc đáo (kể cả nhân vật phụ) và loại bỏ sự trùng lặp mà không làm đứt gãy mạch truyện.

### 3.4. Phân Định Rõ Ràng Lý Do 0 Output (Genuine Zero Reason Distinctions)
- Khi một tập phim hoặc mùa phim không tạo ra video recap nào (0 output), hệ thống phân biệt rạch ròi giữa hai trường hợp:
  1. **Không có kết quả hợp lệ (Genuine Zero)**: Video nguồn không có diễn biến nào khớp với tiêu chí biên tập yêu cầu (ví dụ: tập phim toàn cảnh im lặng, hoặc không có sự kiện nào đạt chuẩn). Trường hợp này được bộ phận kiểm toán độc lập (`verifier`) xác nhận là hợp lệ, báo trạng thái hoàn tất thành công và nêu rõ lý do chính đáng.
  2. **Lỗi kỹ thuật**: Sự cố mạng, lỗi phân tích cú pháp, thiếu dữ liệu phụ đề hoặc lỗi dịch vụ AI. Hệ thống sẽ báo lỗi rõ ràng kèm mã lỗi chi tiết để xử lý.

### 3.5. Không Áp Đặt Chỉ Tiêu Số Lượng Cứng (No Quota)
- Hệ thống **tuyệt đối không áp đặt hạn mức hay chỉ tiêu số lượng nhân tạo** (no quota slicing).
- Nếu nội dung phim có 1, 3 hay nhiều tuyến truyện xuất sắc được chứng minh bằng bằng chứng thực tế, hệ thống sẽ tạo ra bấy nhiêu video recap tương ứng; ngược lại, nếu không có câu chuyện nào đủ chất lượng, hệ thống sẽ thông báo trung thực thay vì cố tình chia nhỏ hay bịa đặt kịch bản để đạt số lượng.

### 3.6. Cơ Chế Hủy & Làm Mới Bộ Nhớ Đệm Thông Minh (Cache Invalidation)
- Khóa bộ nhớ đệm (`cache key`) được tính toán dựa trên mã băm của chính câu nhắc biên tập (`prompt_hash`), cấu hình phân tích và dấu vết của tệp video nguồn.
- Khi người dùng chỉnh sửa câu nhắc trong Cài đặt hoặc thay đổi tệp video, bộ nhớ đệm cũ sẽ tự động được làm mới tương ứng cho các phần bị ảnh hưởng.
- Các tập phim hoặc phân đoạn không thay đổi sẽ tiếp tục tái sử dụng kết quả đã lưu trong bộ nhớ đệm, giúp tiết kiệm thời gian và chi phí API tối đa.

### 3.7. Giới Hạn Thực Tế Về Hình Ảnh & Lời Thoại (Transcript-Only Limitation)
- **Tính chất trung thực**: Hệ thống phân tích kịch bản căn cứ chủ yếu trên phụ đề và lời thoại bóc tách từ video (Transcript-Only).
- **Giới hạn hình ảnh**: Những tình tiết điện ảnh diễn ra hoàn toàn bằng hình ảnh im lặng (như ánh mắt, hành động không lời, hoặc cảnh quay không có hội thoại và không có phụ đề miêu tả) sẽ có giới hạn phản ánh trong kịch bản, trừ khi người dùng bật tùy chọn **"Scanner model supports image/Vision input"** trong Cài đặt để gửi hình ảnh lên mô hình Vision hỗ trợ giải mã.

---

## 4. Các Giai Đoạn Xử Lý (Phases)

Mỗi tập phim và mùa phim được xử lý tuần tự qua các giai đoạn nghiêm ngặt:

1. **`media_probe`**: Thăm dò và phân tích thông số kỹ thuật video (độ phân giải, thời lượng, luồng hình ảnh, luồng âm thanh) bằng FFprobe nhúng.
2. **`subtitles`**: Khám phá và trích xuất phụ đề (ưu tiên tiếng Anh, xử lý tệp phụ đề ngoài hoặc luồng phụ đề nhúng trong container).
3. **`scanner`**: Quét từng phân đoạn transcript/video qua Scanner AI để trích xuất bằng chứng câu chuyện (narrative evidence) gắn mốc thời gian tuyệt đối.
4. **`coverage_check` & `second_pass`**: Đối soát sổ cái độ phủ, phát hiện khoảng trống và quét bổ sung có mục tiêu.
5. **`season_barrier`** (Chế độ Season): Điểm đồng bộ hóa bắt buộc — toàn bộ các tập trong mùa phải hoàn thành Scanner và kiểm tra độ phủ trước khi kết nối.
6. **`season_connecting`** (Chế độ Season): Phân tích mạng lưới liên kết sự kiện, xung đột và nhân vật xuyên suốt toàn bộ các tập của mùa.
7. **`candidate_discovery` & `candidate_consolidation`**: Khám phá các ý tưởng kịch bản tiềm năng và hợp nhất các ứng viên trùng lặp.
8. **`zero_output_verification`**: Kiểm toán độc lập và xác minh kết quả khi số lượng ứng viên bằng 0 hoặc có độ phủ thấp bất thường.
9. **`output_plan_ready`**: Hoàn thiện kế hoạch phân đoạn và kịch bản recap.
10. **Giai đoạn Kết xuất (Rendering)**:
    - Tổng hợp lời dẫn thuyết minh (Timeline narration) bằng mô hình giọng đọc AI.
    - Phối trộn âm thanh thông minh (Real Audio Mix).
    - Tạo tệp phụ đề SRT đồng bộ chính xác đến từng mili-giây.
    - Mã hóa video đầu ra hoàn chỉnh bằng bộ tăng tốc Hybrid GPU hoặc CPU.

---

## 5. Cấu Hình Cài Đặt (Chính Xác 4 Tab)

Nhấn nút **"⚙ Settings"** trên thanh công cụ chính để mở cửa sổ cấu hình trung tâm với đúng 4 tab:

### Tab 1: Recap (Kịch bản & Phim)
- **Recap language**: Ngôn ngữ kịch bản (`en-US`, `en-GB`).
- **Recap mode**: Chế độ recap (`MAIN_STORIES` — tập trung tuyến truyện chính, hoặc `FULL_EPISODE` — tóm tắt toàn diện tập).
- **Content type**: Phân loại nội dung (`US_TV_SHOW`, `DE_GERMAN_SOAP`, `BODYCAM`, `FEATURE_FILM`, `OTHER`).
- **Footage rights**: Trạng thái bản quyền tư liệu (`UNVERIFIED`, `OWNED`, `LICENSED`, `FIRST_PUBLICATION_RIGHTS`, `FAIR_USE`).
- **Recap Prompt**: Khung soạn thảo prompt duy nhất điều khiển toàn bộ pipeline, kèm nút **"Reload Default Prompt"** để khôi phục chỉ dẫn chuẩn của từng thể loại phim.

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

*(Lưu ý: Toàn bộ cơ chế thích ứng tự động và độ tin cậy AI như đo lường byte chính xác, chia nhỏ/cô đọng/hợp nhất đệ quy, khôi phục cache, phase timeouts, số lượt thử lại 3 attempts và phân lô batching được quản lý hoàn toàn tự động ngầm bên trong, không đưa vào giao diện Settings nhằm giữ trải nghiệm người dùng tinh gọn).*

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

## 10. Kiến Trúc Kỹ Thuật (Engineering Architecture)

### 10.1. Nguyên Nhân Gốc Rễ (Root Cause Analysis)
Trước phiên bản v0.4.0 (R8), hệ thống gặp phải các vấn đề cốt lõi sau:
1. **Lỗ hổng độ phủ một lượt quét (Single-pass Coverage Gaps)**: Phân tích Scanner trong một lượt duy nhất dễ bỏ sót các diễn biến quan trọng ở những đoạn hội thoại thưa hoặc cảnh chuyển giao, không có cơ chế đối soát dòng thời gian thực tế.
2. **Ghép nối chỉ dẫn rời rạc (Fragmented Prompt Coupling)**: Các chỉ thị biên tập cho Scanner, Connection và Finalizer bị phân mảnh, thiếu một cơ chế chuyển hóa thống nhất từ một prompt duy nhất của người dùng.
3. **Thiếu giai đoạn tuyển chọn ứng viên (Lack of Candidate Discovery & Consolidation)**: Quá trình chuyển từ bằng chứng sự kiện sang kế hoạch kịch bản thiếu bước khám phá ý tưởng đa chiều và gộp nhóm khử trùng lặp, dễ dẫn đến các kịch bản trùng ý hoặc thiên lệch nhân vật chính.
4. **Không phân định được kết quả 0 output (Ambiguous Zero-Output)**: Khi không có kết quả đầu ra, hệ thống không phân biệt được giữa việc không có câu chuyện phù hợp do nội dung thực tế (Genuine Zero) với các lỗi kỹ thuật hệ thống.

### 10.2. Các Mô-đun Mới Trong Kiến Trúc v0.4.0
Để giải quyết triệt để các nguyên nhân trên, kiến trúc R8 bổ sung các mô-đun chuyên biệt:

1. **`toolrecap_v2.domain.policy` (Editorial Policy Engine)**:
   - Tiếp nhận một câu nhắc duy nhất (`raw_prompt`) từ người dùng và phân tích cú pháp ngoại tuyến thành các chỉ thị cấu trúc:
     - `ScannerDirective`: Quy định danh mục và tiêu chí trích xuất bằng chứng.
     - `CoverageDirective`: Quy định mức độ bao phủ và các khoảng trống cần quét lại.
     - `ConnectionDirective`: Quy định cách thức liên kết sự kiện giữa các tập phim.
     - `CandidateDirective`: Quy định tiêu chí khám phá và lọc ứng viên kịch bản.
     - `OutputDirective` & `ValidationDirective`: Quy định định dạng và ràng buộc tính xác thực.
   - Tạo mã băm chính sách (`policy_hash`) phục vụ cơ chế làm mới bộ nhớ đệm chính xác.

2. **`toolrecap_v2.analyzer.coverage` (Coverage Ledger & Gap Auditor)**:
   - `CoverageLedger`: Theo dõi mốc thời gian chi tiết của từng tập phim, tính toán tỷ lệ bao phủ transcript và bằng chứng.
   - `detect_coverage_gaps`: Tự động nhận diện các khoảng trống thời gian, danh mục hoặc nhân vật thiếu bằng chứng.
   - `reconcile_coverage_gaps` & `plan_second_pass_requests`: Lập kế hoạch và thực hiện quét bổ sung lần 2 (second-pass) có mục tiêu.

3. **`toolrecap_v2.analyzer.candidates` (Candidate Discovery, Consolidation & Verification)**:
   - `discovery.py`: Khám phá ứng viên kịch bản cho tập đơn lẻ (`discover_candidates_single`) và trọn bộ mùa phim (`discover_candidates_season`).
   - `consolidation.py`: Gộp nhóm các ứng viên tương đồng (`consolidate_candidates`), chấm điểm và loại bỏ trùng lặp thông minh.
   - `verifier.py`: Kiểm toán độc lập kết quả khi số lượng ứng viên bằng 0 (`verify_zero_or_low_output`), phân định chính xác 9 mã lý do và xác nhận Genuine Zero hợp lệ (`is_genuine_zero_valid`).

---

## 11. Cập Nhật & Lưu Trữ Dữ Liệu

- **Cập nhật ứng dụng ToolRecap V2**: Tự động kiểm tra GitHub Releases chính thức từ `longthao9820-alt/tool-recap-v2`. Bản cập nhật được xác thực mã băm SHA256 trước khi hoán đổi an toàn có cơ chế khôi phục (rollback).
- **Cập nhật VoiceStudio Subsystem**: Kiểm tra và áp dụng gói cập nhật adapter tách biệt, xác thực mã băm SHA256 và manifest hợp lệ.
- **Nơi lưu trữ dữ liệu người dùng**: Toàn bộ cài đặt, lịch sử hàng đợi và bộ nhớ đệm được lưu tại:
  ```
  %LOCALAPPDATA%\ToolRecapV2
  ```
  - `settings.json`: Cấu hình ứng dụng và API Key (lưu văn bản thuần cục bộ).
  - `projects.json`: Lịch sử và hàng đợi dự án (lưu nguyên tử, chống hỏng hóc khi mất điện).
  - `cache\gateway_analysis\`: Bộ nhớ đệm phân tích AI Gateway theo mã băm video và prompt hash.
  - `cache\episode_evidence\`: Bộ nhớ đệm bằng chứng tập phim và quét phủ lần hai.
  - `cache\subtitles\`: Bộ nhớ đệm phụ đề và kết quả OCR.
  - `voice_subsystem\`: Môi trường runtime Python độc lập cho giọng đọc nâng cao.
  - `models\`: Bộ nhớ đệm trọng số mô hình AI (OCR, STT, TTS).
  - `logs\`: Nhật ký hoạt động và thông báo lỗi.

---

## 12. Đóng Gói Bản Phát Hành Portable

Để tạo bản phân phối Portable độc lập:
1. Nhấp đúp vào `Dong-Goi-ToolRecapV2.cmd` (hoặc chạy lệnh `python build_exe.py`).
2. Kịch bản sẽ tự động:
   - Dựng ứng dụng bằng PyInstaller với tệp cấu hình `ToolRecapV2.spec`.
   - Nhúng FFmpeg, FFprobe, giấy phép và tệp hướng dẫn sử dụng vào `release\ToolRecapV2\`.
   - Chạy kiểm tra tự động `--version` và `--self-check` trên tệp thực thi đã dựng.
   - Nén toàn bộ thành `release\ToolRecapV2-v0.4.0-windows-portable.zip` và tạo tệp mã băm companion `ToolRecapV2-v0.4.0-windows-portable.zip.sha256.txt`.

---

## 13. Giấy Phép Bản Quyền (Licenses)

- Mã nguồn chính của ToolRecap V2 được phát hành theo giấy phép **MIT License**.
- Các thành phần bên thứ ba (FFmpeg, Piper, RapidOCR, OmniVoice, faster-whisper, PyTorch, OpenCV, Shapely) tuân theo giấy phép mã nguồn mở tương ứng. Xem chi tiết tại tệp `THIRD_PARTY_LICENSES.md`.
- Người dùng tự chịu trách nhiệm về bản quyền của video nguồn và việc sử dụng các mô hình AI theo điều khoản của nhà cung cấp.
