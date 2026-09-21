# ToolRecap V2

ToolRecap V2 automates the proven workflow:

```text
SELECT VIDEO / SELECT FOLDER
  -> Source Preparation
  -> Scanner
  -> Finalizer + raw Recap Prompt
  -> Final JSON
  -> Technical Validation / bounded AI repair
  -> Voice + deterministic render
  -> Finished videos
```

The configured AI Finalizer is the editor. ToolRecap prepares sources, transports
Scanner observations, validates the Final JSON technically, and executes that plan.
It does not rank candidates, choose stories, impose output quotas, or rewrite the
Finalizer's editorial decisions.

## Project selection

- **Select File** creates one `SINGLE_EPISODE` project.
- **Select Folder** discovers supported videos directly in that folder, naturally
  sorts them, and creates one `SEASON` project containing every discovered source.
- One Start action runs the complete workflow. Manual JSON import is not part of
  the normal production path.

Supported source extensions are `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`,
and `.ts`.

## AI Gateway settings

The Gateway is OpenAI-compatible and provider-neutral. ToolRecap sends the exact
configured role names; no code path depends on a particular provider or model name.

- Enable AI Gateway analysis
- API endpoint and API key
- Scanner model and thinking level
- Scanner Vision support
- Finalizer model and thinking level
- Scanner parallelism
- Scanner chunk length
- Test Scanner / Test Finalizer

The Scanner extracts grounded, timestamped source observations. It does not decide
which stories become outputs. The Finalizer receives the complete project source
mapping, Scanner observations, renderer contract, and the user's raw Recap Prompt.
It returns one canonical Final JSON with zero, one, or any number of outputs.

Scanner requests retain their 500 KB per-chunk safeguard. Whole-project Finalizer
requests do not inherit that limit: ToolRecap losslessly packs repeated transport
structure, measures/logs the request, and lets the configured Gateway/model enforce
its real context capacity. No observation, timestamp, or episode is truncated to fit
an application-side byte quota.

## Final JSON boundary

Every output contains an ID, title, file/output metadata, language, and ordered
segments. Segments contain their type, purpose, narration/dialogue text, audio and
subtitle policies, visual-speed recommendation, and source clips. Clips identify an
episode/source plus exact millisecond timestamps.

Technical validation checks:

- required root, output, segment, and clip fields;
- unique output and segment IDs;
- supported audio policies;
- exact episode/source resolution;
- `0 <= start_ms < end_ms <= source duration`;
- source existence for prepared projects;
- narration requirements and renderer executability.

Validation never scores story quality. If validation fails, the exact structured
errors are sent back to the configured Finalizer for a bounded repair. ToolRecap
does not clamp timestamps or silently substitute footage.

`{"outputs": []}` is an explicit, valid editorial result. Missing `outputs`, malformed
JSON, transport errors, source errors, and timestamp errors remain technical errors;
they are never converted into “no candidates.”

## Cache and resume

Cache dependencies are layered:

- source extraction depends on source identity and extraction settings;
- Scanner evidence depends on source identity plus Scanner model/thinking/Vision/
  chunk settings;
- Final JSON depends on Scanner evidence, Finalizer model/thinking, raw Recap Prompt,
  and recap metadata;
- render output depends on Final JSON plus voice/audio/encoder/render settings.

A Finalizer retry reuses Scanner evidence. Voice or render changes reuse Final JSON.
Prompt or Finalizer changes regenerate Final JSON while retaining compatible source
and Scanner artifacts. A failed multi-output render validates and skips previously
completed outputs when its render dependency signature is unchanged.

## Renderer and publication output

The existing V2 renderer remains responsible for local voice generation, source
cutting, original-audio preservation/mixing, narration and original-dialogue SRT,
GPU encoding, cancellation, and final output validation. It executes every valid
Finalizer output without candidate filtering.

Each publication folder contains exactly:

```text
<title>.mp4
<title>.narration.srt
<title>.original.srt
```

## Bộ kết xuất đa nguồn mạnh mẽ (Robust Multi-Source Renderer)

Bộ kết xuất phiên bản v0.5.2 được nâng cấp với khả năng xử lý đa nguồn mạnh mẽ, đảm bảo tính toàn vẹn và mượt mà cho mọi video đầu ra:

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

## Updates

Releases contain a Windows portable ZIP and SHA256 companion file. The updater:

1. fetches release metadata;
2. downloads and verifies SHA256;
3. safely extracts and validates staging;
4. creates and verifies a backup before touching the installation;
5. applies and verifies the new executable;
6. rolls back from the verified backup on failure;
7. restarts ToolRecap.

Settings, caches, downloaded voice models, and project state live outside the app
directory under `%LOCALAPPDATA%\ToolRecapV2` and survive routine updates.

## Development

```powershell
python -m pip install -r requirements.txt
pytest -q
python build_exe.py
```

The build creates a portable directory, ZIP, SHA256 file, and performs executable
`--version` and `--self-check` checks before reporting success.

See `ARCHITECTURE_REFACTOR_REPORT.md` for the implementation audit, code paths,
test evidence, and known limitations.
