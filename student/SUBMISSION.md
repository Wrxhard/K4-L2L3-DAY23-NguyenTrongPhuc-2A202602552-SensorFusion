# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên: Nguyễn Trọng Phúc
- MSSV: 2A202602552
- Email: phucmxst123@gmail.com (theo cấu hình git; cần học viên xác nhận)
- Link repo (fork): https://github.com/Wrxhard/K4-L2L3-DAY23-NguyenTrongPhuc-2A202602552-SensorFusion
- Commit code E–H đã kiểm tra: `850c2d8` (CP4).
- Commit hash nộp (`git rev-parse HEAD`): Chưa có bản nộp cuối; CP5 còn thiếu dữ liệu Waymo.

## Trạng thái checkpoint

| Checkpoint | Kết quả thực tế | Commit |
|---|---|---|
| CP0 | Môi trường và CLI hoạt động, cấu hình local frame 0–198; còn thiếu segment Waymo | `12100e6` |
| CP1 — EKF | 4 test Part E pass | `b9e71c2` |
| CP2 — Camera | 20 test theo bộ lọc CP2 pass | `6abb1b1` |
| CP3 — Association | 9 test theo bộ lọc CP3 pass | `cea751b` |
| CP4 — Lifecycle | Toàn bộ 128 test pass; không còn failed/xfailed | `850c2d8` |
| CP5 — Waymo | Chưa hoàn tất: thiếu segment mặc định; Drive của lớp báo vượt hạn mức tải | Chưa có |
| CP6 — Báo cáo và nộp | Đã chuẩn bị giải thích và khai báo AI; chưa đủ artifacts để nộp | Chưa hoàn tất |

Lệnh kiểm tra từ gốc repo trên Windows:

```powershell
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe -m pytest student/tests -q --disable-warnings
```

Kết quả sau CP4: `128 passed, 830 warnings in 10.74s`. Warnings chủ yếu là cảnh báo `np.matrix` từ API của đề. Giữ nguyên platform và test. `PYTHONUTF8=1` giúp các test tạo báo cáo tiếng Việt trên Windows.

## Tóm tắt kết quả

**Chưa có số liệu Waymo; báo cáo chưa sẵn sàng nộp.** Không tạo hoặc sửa tay artifacts. Chỉ so sánh hai mode khi đã chạy thật.

Weights FPN đã tải từ [SFA3D chính thức](https://github.com/maudzung/SFA3D/tree/master/checkpoints/fpn_resnet_18) được `data/README.md` dẫn; `create_fpn_model` đã nạp thành công. Lệnh compare đã thử nhưng dừng với `FileNotFoundError` vì thiếu segment mặc định trong `data/Waymo/`. ZIP `Day23-Fusion-...-003.zip` có sẵn chỉ chứa weights Darknet, không có Waymo. Các file dữ liệu/weights/ZIP vẫn nằm local, không đưa vào commit.

- `fusion_mode`, `frames`, `segment`, `seed`: Cấu hình dự kiến `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`; chưa phải kết quả chạy.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: Chưa đo.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: Chưa đo.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: Chưa đo.
- Giải thích khác biệt hai mode: Chưa đủ bằng chứng để kết luận camera cải thiện hay làm xấu tracking.

Khi có dữ liệu, chạy từ gốc repo:

```powershell
$env:PYTHONUTF8 = '1'
$env:DAY23_HEADLESS = '1'
.venv/Scripts/python.exe -m fusion_lab.scripts.run_lab --config student/config/paths.yaml --fusion compare --seed 0
.venv/Scripts/python.exe tools/check_submission.py
```

Cần đối chiếu `rmse = sqrt(sum_sq_err / matches)` trên vị trí 3D của confirmed tracks ghép một-một với GT trong cổng XY 2 m; RMSE null nếu không có cặp. Đọc RMSE cùng matches, ghost và miss: ít cặp ghép có thể cho RMSE thấp nhưng bỏ sót nhiều xe. Tính `precision_track = matches / (matches + ghost_track_frames)` và `coverage = matches / det_tp`. Mục tiêu đủ điểm là RMSE mỗi mode ≤ 0.45 m, precision_track ≥ 0.75, coverage ≥ 0.70 và `rmse_fused - rmse_lidar ≤ 0.05 m`.

Mỗi `(mode, frame)` phải có đúng một record log, với `matches + ghosts == confirmed` và `matches + misses == valid_gt`. Tổng/trung bình log phải khớp metrics; detection phải giống nhau giữa hai mode.

Camera lab lấy tâm hộp 2D ground-truth FRONT rồi thêm nhiễu seeded, không chạy detector ảnh. Kết quả fused không chứng minh chất lượng một camera detector hoặc perception system độc lập với GT.

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

Dẫn chứng: `run_single` trong [run_lab.py](../platform/fusion_lab/scripts/run_lab.py) gọi vòng `KF.predict(track)`, sau đó `associate_and_update(..., lidar_sensor)`, rồi `associate_and_update(..., camera_sensor)`; không predict giữa hai update. `grade_run.log` chỉ ghi kết quả tổng hợp từng mode/frame, không ghi thứ tự update nội bộ. Chưa chạy Waymo nên chưa có record log thực tế để viện dẫn; thứ tự trên được xác minh từ code và test.

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

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): OpenAI Codex trong ChatGPT.
- Dùng cho phần nào (hàm, câu hỏi, debug): Đọc yêu cầu; triển khai tất cả hàm Part E–H; cài môi trường và kiểm tra; chuẩn bị phần giải thích báo cáo, tìm dữ liệu/weights.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Codex chạy test gốc trước triển khai (82 passed, 46 xfailed), đối chiếu công thức với stub/hướng dẫn kỹ thuật, chạy test từng checkpoint rồi toàn bộ 128 test sau CP4. Chưa chạy Waymo thành công hoặc đối chiếu metrics/log. Học viên cần tự đọc, kiểm tra và giải thích được code/báo cáo khi vấn đáp.

## Checklist nộp

- [x] Part E–H đã implement; toàn bộ 128 test pass, không failed/xfailed.
- [x] Đã đọc và giữ nguyên Part A–D, platform và tests.
- [ ] Chạy chấm điểm compare, seed 0, frames 0–198 thành công.
- [ ] Đã commit đủ 6 artifacts do pipeline sinh, không sửa tay.
- [ ] Đã điền số liệu thực tế, so sánh hai mode và commit hash nộp cuối.
- [x] Có khai báo AI phản ánh đúng việc đã thực hiện.
- [x] Không commit Waymo, weights, paths.yaml, ZIP hoặc API key.
- [ ] check_submission.py báo sẵn sàng nộp.
- [ ] Đã push và nộp link repo + commit hash trên LMS.
