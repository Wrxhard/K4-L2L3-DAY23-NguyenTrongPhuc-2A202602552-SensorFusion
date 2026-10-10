# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên: Nguyễn Trọng Phúc
- MSSV: 2A202602552
- Email: phucmxst123@gmail.com (theo cấu hình git)
- Link repo (fork): https://github.com/Wrxhard/K4-L2L3-DAY23-NguyenTrongPhuc-2A202602552-SensorFusion
- Commit code E–H đã kiểm tra: `850c2d8` (CP4).
- Commit hash nộp (`git rev-parse HEAD`): Bản nộp cuối là HEAD sau CP6; lấy hash 40 ký tự bằng lệnh trên. Hash cuối được cung cấp cùng link repo khi bàn giao.

## Trạng thái checkpoint

| Checkpoint | Kết quả thực tế | Commit |
|---|---|---|
| CP0 | Môi trường và CLI hoạt động; data/ đã đủ Waymo và weights, cấu hình frame 0–198 | `12100e6` |
| CP1 — EKF | 4 test Part E pass | `b9e71c2` |
| CP2 — Camera | 20 test theo bộ lọc CP2 pass | `6abb1b1` |
| CP3 — Association | 9 test theo bộ lọc CP3 pass | `cea751b` |
| CP4 — Lifecycle | Toàn bộ 128 test pass; không còn failed/xfailed | `850c2d8` |
| CP5 — Waymo | Compare seed 0, đủ 199 frame/mode; 6 artifacts đã kiểm tra và commit | `b6be483` |
| CP6 — Báo cáo | Số liệu thực tế, 6 câu giải thích, khai báo AI và kiểm tra nộp bài | Commit cuối có tiền tố `CP6:` |

Lệnh kiểm tra từ gốc repo trên Windows:

```powershell
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe -m pytest student/tests -q --disable-warnings
```

Kết quả sau CP4: `128 passed, 830 warnings in 10.74s`. Warnings chủ yếu là cảnh báo `np.matrix` từ API của đề. Giữ nguyên platform và test. `PYTHONUTF8=1` giúp các test tạo báo cáo tiếng Việt trên Windows.

## Tóm tắt kết quả

Nguồn: [metrics.json](artifacts/metrics.json) và [grade_run.log](artifacts/grade_run.log), do pipeline sinh ra; không sửa tay artifacts.

- `fusion_mode`: `compare`.
- `frames`: `[0, 198]` (199 frame mỗi mode, tính cả hai đầu).
- `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- `seed`: `0`.
- `detection.precision`: `0.9700934579439252`; `detection.recall`: `0.7004048582995951`.
- `detection.tp/fp/fn`: `519/16/222`; detection giống nhau giữa hai mode ở mọi frame.

| Tracking metric | LiDAR | Fused |
|---|---:|---:|
| RMSE (m) | 0.15032268781360134 | 0.17109183025633565 |
| matches | 502 | 502 |
| sum_sq_err (m²) | 11.343649056695735 | 14.694752018992311 |
| ghost_track_frames | 0 | 0 |
| missed_gt_frames | 239 | 239 |
| mean_confirmed_tracks | 2.522613065326633 | 2.522613065326633 |
| precision_track | 1.0 | 1.0 |
| coverage = matches / det_tp | 0.9672447013487476 | 0.9672447013487476 |

Trong lần chạy này, camera làm RMSE tăng `0.020769142442734312 m`, khoảng 2,08 cm, trong khi matches, ghost, miss và mean confirmed không đổi. Fused không cải thiện sai số vị trí, nhưng vẫn đạt RMSE ≤ 0.45 m và chênh lệch ≤ 0.05 m. Hai mode đều có precision_track = 1 và coverage ≈ 96,72%, vượt ngưỡng 0.75 và 0.70. Vì số cặp ghép giống nhau, khác biệt RMSE không đến từ việc đánh giá ít cặp hơn; sum_sq_err tăng từ 11.343649056695735 lên 14.694752018992311 m².

Ví dụ frame 46: mỗi mode có 2 matches, 0 ghost, 1 miss; sum_sq_err LiDAR = 0.04450990452758851 m², fused = 0.04779304775963698 m². Frame 0 có 2 detection TP nhưng 0 confirmed track, 2 miss ở cả hai mode: tracker cần tích luỹ score trước khi xác nhận. Frame 198 có 9 GT, 3 detection TP, 6 detection FN; tracking có 3 matches, 0 ghost, 6 miss. Vì vậy phải đọc cả miss và detection, không chỉ RMSE thấp hoặc ghost bằng 0. Tổng 239 tracking miss và 222 detection FN chênh 17; đây là chênh lệch tổng số đếm, không phải phân rã nguyên nhân theo từng xe.

Nhiễu pixel và việc tâm hộp 2D không nhất thiết trùng phép chiếu tâm hộp 3D có thể làm camera update không cải thiện state. Chưa có ablation để xác định mức đóng góp từng yếu tố, vì vậy không khẳng định calibration sai. Camera lab dùng tâm hộp ground-truth FRONT rồi thêm nhiễu seeded, không chạy detector ảnh; kết quả không chứng minh chất lượng camera detector hoặc perception system độc lập với GT.

Đã kiểm tra 199 record mỗi mode, 398 record compare, không thiếu/trùng `(mode, frame)`. Mọi record thoả `matches + ghosts == confirmed` và `matches + misses == valid_gt`. Đã tính lại metrics từ log bằng `validate_metrics_records` cho compare và hai file per-mode; log per-mode khớp chính xác phần tương ứng trong log gộp. `rmse = sqrt(sum_sq_err / matches)` trên vị trí 3D của confirmed tracks ghép một-một với GT trong cổng XY 2 m.

Lệnh tái lập từ gốc repo trên Windows:

```powershell
$env:PYTHONUTF8 = '1'
$env:DAY23_HEADLESS = '1'
.venv/Scripts/python.exe -m fusion_lab.scripts.run_lab --config student/config/paths.yaml --fusion compare --seed 0
.venv/Scripts/python.exe tools/check_submission.py
```

Dữ liệu và weights đặt local trong `data/`; `student/config/paths.yaml` trỏ tới segment trên, frame_start 0, frame_end 198. Không commit dữ liệu, weights, config local hoặc ZIP. Giữ nguyên Part A–D, platform và test gốc.

## Giải thích ngắn (Parts E–H)

### 1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

State chung `x = (px, py, pz, vx, vy, vz)^T`. LiDAR đo vị trí `z = (x_s, y_s, z_s)^T` trong hệ cảm biến, đơn vị mét; R kích thước 3×3, đường chéo là bình phương sigma LiDAR. Mô hình tuyến tính theo vị trí, `H = [R_vehicle_to_sensor, 0]`.

Camera đo `z = (u, v)^T` bằng pixel; R kích thước 2×2, đường chéo `sigma_cam_i²`, `sigma_cam_j²` (mặc định 25 pixel²). Sau đổi hệ `p_s = R p + t`, pinhole cho `u = c_i - f_i*y_s/x_s`, `v = c_j - f_j*z_s/x_s`. Hàm này phi tuyến, platform tính Jacobian tại state hiện tại; camera không đo trực tiếp độ sâu. Cả hai dùng `gamma = z - h(x)`, `S = H P H^T + R`, nhưng khác số chiều và đơn vị.

Dẫn chứng: `innovation`, `innovation_covariance`, `ekf_update` trong [kalman.py](workspace/kalman.py); `camera_measurement_prediction`, `build_camera_measurement` trong [camera_fusion.py](workspace/camera_fusion.py).

### 2. Vì sao cần gating Mahalanobis trước khi gán?

Euclid không phản ánh độ bất định state và noise từng sensor. `d² = gamma^T S^-1 gamma` chuẩn hoá residual theo covariance innovation; cổng `chi2.ppf(gating_threshold, dim_meas)` giữ cặp tương thích với mô hình. Mặc định xác suất 0.995, bậc tự do 3 cho LiDAR và 2 cho camera. Greedy vì vậy không buộc ghép một đo xa khi mọi cặp đều không hợp lệ.

[association.py](workspace/association.py) kiểm tra FOV trước khoảng cách, ghi inf cho cặp bị loại, chọn cặp hữu hạn nhỏ nhất rồi xoá hàng/cột để gán một-một. Dùng `np.linalg.solve` thay cho nghịch đảo tường minh. Gating giảm gán sai nhưng không bảo đảm tránh mọi lần đổi ID khi xe gần nhau.

### 3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log/code.

Đây là **track-then-fuse**: một danh sách track, predict một lần/frame, gán/update LiDAR và quản lý vòng đời, rồi gán/update camera lên cùng track. Camera không tạo tracker riêng hoặc gộp raw image với point cloud trước detector. Detector cho hộp từng frame; tracker duy trì ID, vị trí, vận tốc và covariance qua thời gian.

Dẫn chứng: `run` trong [run_lab.py](../platform/fusion_lab/scripts/run_lab.py) gọi vòng `KF.predict(track)`, sau đó `associate_and_update(..., lidar_sensor)`, rồi `associate_and_update(..., camera_sensor)`; không predict giữa hai update. `grade_run.log` chỉ ghi kết quả tổng hợp từng mode/frame, không ghi thứ tự update nội bộ; thứ tự được xác minh từ code và test.

### 4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

Extrinsic sai làm sai p_s và pixel dự đoán dù LiDAR đúng. Innovation camera có thể có bias theo một hướng hoặc thay đổi theo vị trí/độ sâu, thay vì dao động quanh 0. Khi residual lớn so với S, Mahalanobis vượt cổng χ² và đo bị loại. Khi P lớn làm S lớn, hoặc lệch nhỏ vẫn trong cổng, EKF có thể nhận đo sai, kéo state lệch và tăng RMSE fused. FOV cũng có thể bị tính sai.

Cần xem innovation, tỷ lệ cặp bị loại và RMSE/ghost/miss; một residual lớn chưa đủ kết luận calibration sai. Đây là giải thích từ mô hình `camera_measurement_prediction` và `mahalanobis_distance`, chưa phải kết quả thực nghiệm calibration.

### 5. Vì sao cần sensor tường minh ở frame rỗng? Vì sao lidar quyết định score/init/delete?

Khi `meas_list=[]`, không thể lấy sensor từ phần tử đầu. Bỏ qua lượt đó sẽ khiến track mất đo LiDAR không giảm score và tồn tại sai. [associate_and_update](workspace/association.py) luôn kết thúc bằng `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`, kể cả không có đo.

[TrackManager](../platform/fusion_lab/tracking/manager.py) chỉ cộng score hit, trừ score miss trong FOV, tạo và xoá track ở lượt LiDAR. Camera chỉ cập nhật EKF trên track được ghép; camera rỗng/ngoài FOV/không ghép được không đổi score và trạng thái tồn tại. Quy tắc tránh đếm hai lần bằng chứng tồn tại trong một frame và không phạt xe ngoài FOV camera. Sensor tường minh giữ đúng hành vi ở cả lượt LiDAR rỗng và lượt camera rỗng.

### 6. Điều kiện xác nhận, giữ confirmed sau miss và xoá track?

[track_management.py](workspace/track_management.py) khởi tạo vị trí qua sens_to_veh, vận tốc 0, covariance vị trí `R_rotation R_meas R_rotation^T`, covariance vận tốc `diag(sigma_p44², sigma_p55², sigma_p66²)`. Ban đầu state initialized, score 1/window (mặc định 1/6).

Hit LiDAR tăng 1/6, tối đa 1; track chưa confirmed chuyển tentative khi hit và confirmed khi **score > 0.8**, không dùng >=. Miss trong FOV LiDAR giảm 1/6. Track confirmed giữ nhãn confirmed sau miss; ví dụ score 1 giảm còn 5/6 nên vẫn confirmed và chưa xoá. Ngoài FOV LiDAR không trừ score miss.

Xoá trong lượt LiDAR nếu một điều kiện đúng: Pxx > 9 hoặc Pyy > 9; confirmed có score < 0.6; chưa confirmed có score <= 0. Tại đúng variance 9 hoặc score confirmed 0.6, điều kiện tương ứng chưa kích hoạt. Camera không tạo, xoá hoặc đổi score. Các biên được test trong `test_tracking_contract.py` và `test_tracking_regressions.py`.

## Bonus (không bắt buộc)

- Không. Chưa có bằng chứng CVAT, ảnh trực quan hoặc thử nghiệm calibration.

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Codex (ChatGPT).
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ triển khai Part E–H, thiết lập môi trường, debug và soạn báo cáo. Học viên tự chuẩn bị và giải nén dữ liệu Waymo.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Đối chiếu công thức trong hướng dẫn; 128 test pass; chạy Waymo compare seed 0, frames 0–198 và kiểm tra metrics khớp log.

## Checklist nộp

- [x] Part E–H đã implement; toàn bộ 128 test pass, không failed/xfailed.
- [x] Đã đọc và giữ nguyên Part A–D, platform và tests.
- [x] Chạy chấm điểm compare, seed 0, frames 0–198 thành công.
- [x] Đã commit đủ 6 artifacts do pipeline sinh, không sửa tay.
- [x] Đã điền số liệu thực tế, so sánh hai mode và hướng dẫn lấy commit hash cuối.
- [x] Có khai báo AI phản ánh đúng việc đã thực hiện.
- [x] Không commit Waymo, weights, paths.yaml, ZIP hoặc API key.
- [x] check_submission.py xác nhận sẵn sàng nộp sau commit CP6.
- [ ] Học viên nộp link repo và hash HEAD cuối trên LMS; thao tác này riêng với việc hoàn thiện repo.
