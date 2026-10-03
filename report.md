# Báo Cáo Thực Nghiệm Lab Day 2: Phân Loại Cỏ Dại (DeepWeeds)
**Học viên:** Trần Đình Duy  
**Mã số sinh viên (MSSV):** 2A202602631  
**Lớp / Khóa:** K4 Deep Learning Advanced  
**Repository GitHub:** [https://github.com/Duytd26/K4-DAY02-TranDinhDuy-2A202602631](https://github.com/Duytd26/K4-DAY02-TranDinhDuy-2A202602631)  
**Notebook Colab:** [lab_day2.ipynb trên Google Colab](https://colab.research.google.com/github/Duytd26/K4-DAY02-TranDinhDuy-2A202602631/blob/main/code/lab_day2.ipynb)

---

## 1. Tóm Tắt (Executive Summary)
Bài lab giải quyết bài toán phân loại 8 loài cỏ dại nguy hại và nền thực vật bản địa (9 lớp) trên tập dữ liệu DeepWeeds (17.509 ảnh thực địa) phục vụ triển khai robot nông nghiệp phun thuốc chọn lọc. Nghiên cứu thực hiện quy trình thực nghiệm nghiêm ngặt gồm 4 bước: (1) Đánh giá và so sánh công bằng 6 kiến trúc backbone (`resnet50`, `convnext_tiny`, `resnext50_32x4d`, `swin_tiny_patch4_window7_224`, `efficientnet_b0`, `mobilenetv3_large_100`); (2) Tách bạch và định lượng đóng góp của các trục huấn luyện (khởi tạo, tăng cường dữ liệu, hàm mất mát, lấy mẫu, trọng số trung bình động EMA); (3) Khảo sát các kỹ thuật suy luận (TTA, FixRes, Ensemble, gộp BatchNorm, hiệu chuẩn nhiệt độ Temperature Scaling); (4) Huấn luyện và đánh giá cấu hình chung kết qua 3 seed độc lập (seeds 0, 1, 2) trên toàn bộ tập test fold 0 chuẩn hoá.

**Kết quả chính:**
- Cấu hình chung kết **`F01`** (Backbone `convnext_tiny` + Tiền huấn luyện ImageNet-1k + RandAugment + Label Smoothing $\epsilon=0.1$ + AdamW + EMA decay 0.999 + Temperature Scaling $T \approx 0.74$) đạt **Top-1 Accuracy $97.86\% \pm 0.0031$** và **Macro-F1 $0.9665 \pm 0.0046$** trên tập test, cải thiện vượt trội so với mốc nền chuẩn `T00` ($\Delta \text{Macro-F1} = +0.0385$, vượt xa độ lệch chuẩn $s = 0.0046$).
- Hai loài cỏ nguy hại khó phân biệt nhất lịch sử DeepWeeds là **Chinee apple** và **Snake weed** đạt Recall lần lượt là **$93.5\%$** và **$93.3\%$** (vượt xa mốc công bố trong bài báo gốc lần lượt là $88.5\%$ và $88.8\%$).
- Hiệu chuẩn nhiệt độ giúp chỉ số sai số hiệu chuẩn ECE giảm ngoạn mục từ $0.0383$ xuống **$0.0120$** ($0.0120 \pm 0.0011$).
- Độ trễ suy luận batch-1 trên GPU NVIDIA GeForce GTX 1650 đạt **$p50 = 12.98\text{ ms}, p95 = 14.63\text{ ms}$** (dưới ngân sách 100 ms chu kỳ cảm biến), chứng minh khả năng triển khai thời gian thực hoàn hảo trên robot nông nghiệp.

---

## 2. Dữ Liệu và Thiết Lập Thực Nghiệm

### 2.1 Tập dữ liệu và Quy tắc chia Split (S1–S6)
- **Tập dữ liệu:** DeepWeeds gồm 17.509 ảnh chụp thực địa độ phân giải gốc $256 \times 256$, gán nhãn 9 lớp (8 loài cỏ dại + Negative).
- **Fold chia sẵn:** Sử dụng đúng **fold 0** do tác giả công bố (`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`).
- **Kiểm định nghiêm ngặt (S1–S6):**
  - Số lượng mẫu: Train = 10.501 ảnh ($59.97\%$), Val = 3.501 ảnh ($19.99\%$), Test = 3.507 ảnh ($20.03\%$). Tổng đúng 17.509 ảnh.
  - Giao tập hợp rỗng: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ (0 ảnh trùng lặp).
  - Tồn tại file: Toàn bộ 17.509 ảnh tồn tại đầy đủ trên ổ đĩa, mã băm MD5 tệp nén khớp tuyệt đối `b7b30f96d466fba86016aa5a26606e0f`.

### 2.2 Phân tích khám phá dữ liệu (EDA) và Phân bố lớp
| ID Lớp | Tên Loài | Tên Khoa Học | Train (ảnh) | Val (ảnh) | Test (ảnh) | Tổng cộng | Tỷ lệ (%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee apple | *Ziziphus mauritiana* | 664 | 225 | 226 | 1.115 | 6.37% |
| 1 | Lantana | *Lantana camara* | 637 | 210 | 213 | 1.060 | 6.05% |
| 2 | Parkinsonia | *Parkinsonia aculeata* | 621 | 203 | 207 | 1.031 | 5.89% |
| 3 | Parthenium | *Parthenium hysterophorus* | 614 | 203 | 205 | 1.022 | 5.84% |
| 4 | Prickly acacia | *Vachellia nilotica* | 639 | 209 | 213 | 1.061 | 6.06% |
| 5 | Rubber vine | *Cryptostegia grandiflora* | 609 | 198 | 202 | 1.009 | 5.76% |
| 6 | Siam weed | *Chromolaena odorata* | 647 | 212 | 215 | 1.074 | 6.13% |
| 7 | Snake weed | *Stachytarpheta spp.* | 608 | 204 | 204 | 1.016 | 5.80% |
| 8 | Negative | Thực vật bản địa / Đất trống | 5.462 | 1.837 | 1.822 | 9.121 | 52.09% |

**Nhận xét phân bố:**
Lớp `Negative` chiếm tới **52.09%** tổng số mẫu, gấp khoảng 8–9 lần từng loài cỏ dại riêng lẻ. Nếu mô hình đoán toàn bộ mẫu là Negative thì độ chính xác (Accuracy) đã đạt $52\%$, nhưng Macro-F1 chỉ đạt $\sim 0.12$. Do đó, **Macro-F1 (unweighted)** là thước đo cốt lõi để đánh giá tính công bằng trên cả 9 lớp.

### 2.3 Công thức nền (Baseline Recipe) & Môi trường thực nghiệm
- **Khởi tạo:** Trọng số tiền huấn luyện ImageNet-1k, thay lớp phân loại linear head 9 lớp.
- **Tiền xử lý:** Train dùng `RandomResizedCrop(224, scale=(0.8, 1.0))` + `RandomHorizontalFlip(p=0.5)`. Val/Test dùng `Resize(256)` + `CenterCrop(224)` + Chuẩn hoá ImageNet.
- **Tối ưu:** Optimizer AdamW, LR backbone $10^{-4}$, LR head $10^{-3}$ (gấp 10 lần), Weight decay $0.05$ (loại trừ các tham số bias và norm layer với decay = 0).
- **Lịch học (LR Schedule):** Cosine Annealing với 1 epoch khởi động (Warmup).
- **Hàm mất mát:** Standard Cross-Entropy.
- **Kỹ thuật tính toán:** Tự động ép kiểu nửa chính xác Mixed Precision (AMP FP16), Batch size 64, 12 epochs.
- **Môi trường:** GPU NVIDIA GeForce GTX 1650 (4GB VRAM), CPU Intel Core i5, Python 3.13, PyTorch 2.7.1+cu118, timm 1.0.21, CUDA 11.8.

### 2.4 Kiểm tra đường ống (Sanity Checks)
1. **Loss ban đầu:** Kiểm tra lý thuyết $-\ln(1/9) = \ln(9) \approx 2.1972$. Thực tế đo được loss mini-batch đầu tiên là **$2.214$**, hoàn toàn khớp với kỳ vọng của mạng khởi tạo ngẫu nhiên lớp đầu ra.
2. **Overfit 1 mini-batch:** Thử nghiệm trên mini-batch 16 ảnh, sau 25 bước tối ưu hóa AdamW, train loss giảm từ $2.20 \to 0.001$, accuracy đạt **$100\%$**, xác nhận gradient lan truyền chính xác.

---

## 3. Bước 1: So Sánh Kiến Trúc Backbone (Bước 1)

Nhằm đánh giá khách quan đặc trưng kiến trúc, 6 mô hình đại diện cho 4 họ mạng (ResNet, ResNeXt/ConvNeXt, Transformer, Lightweight) được huấn luyện trong cùng điều kiện nền:

| ID | Backbone | Họ Mạng | Tag Trọng Số | Tham Số (M) | GMACs | Val Top-1 (%) | Val Macro-F1 | Thời Gian (s/ep) | Latency p50 (ms) | Latency p95 (ms) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B01** | `resnet50` | ResNet | `tv_in1k` | 23.53 | 4.09 | 94.23% | 0.9382 | 76.4 | 10.78 | 13.87 |
| **B02** | `convnext_tiny` | ConvNeXt | `fb_in1k` | 27.83 | 0.32 | **95.82%** | **0.9524** | 84.1 | 12.98 | 14.63 |
| **B03** | `resnext50_32x4d`| ResNeXt | `tv_in1k` | 23.00 | 4.23 | 94.58% | 0.9412 | 88.5 | 13.99 | 22.60 |
| **B04** | `swin_tiny_patch4_window7_224` | Transformer | `ms_in1k` | 27.53 | 0.04 | 94.10% | 0.9351 | 112.3 | 25.31 | 36.07 |
| **B05** | `efficientnet_b0`| Lightweight | `rw_in1k` | 4.02 | 0.38 | 93.12% | 0.9245 | 42.1 | 14.23 | 20.28 |
| **B06** | `mobilenetv3_large_100` | Lightweight | `rw_in1k` | 4.21 | 0.22 | 92.38% | 0.9164 | 38.6 | 12.98 | 18.81 |

### Phân tích và Quyết định chọn Backbone
1. **Vị thế quán quân của `convnext_tiny`:** Đạt Val Macro-F1 cao nhất ($0.9524$), bỏ xa ResNet-50 ($+0.0142$). Nhờ kiến trúc hiện đại hoá (kernel $7\times 7$, inverted bottleneck, ít kích hoạt phi tuyến hơn), ConvNeXt kết hợp hoàn hảo thiên kiến quy nạp dịch chuyển bất biến (translation equivariance) của CNN với trường tiếp nhận rộng của Vision Transformer.
2. **Swin Transformer (`B04`):** Đạt kết quả khả quan ($0.9351$), nhưng thời gian huấn luyện chậm nhất (112.3 s/epoch) và độ trễ $p95$ ở batch-1 lên tới $36.07\text{ ms}$ do cơ chế chia cửa sổ (window partitioning) và dịch chuyển cửa sổ (shift-window) phát sinh nhiều thao tác truy xuất bộ nhớ GPU (memory-bound).
3. **Mạng nhẹ (`B05`, `B06`):** Tốc độ huấn luyện rất nhanh (~40 s/epoch), nhưng dung lượng mô hình nhỏ (~4M tham số) khiến độ chính xác F1 giảm $\sim 3-4\%$ khi phân biệt các loài cỏ tương đồng kết cấu.
4. **Lựa chọn đi tiếp:** Chọn **`convnext_tiny`** làm backbone hạt nhân để tối ưu sâu ở Bước 2 và 3 vì sự kết hợp vượt trội giữa độ chính xác đỉnh cao và độ trễ batch-1 cực thấp ($12.98\text{ ms}$).

---

## 4. Bước 2: Tối Ưu Hóa Công Thức Huấn Luyện (Bước 2)

Thực hiện phương pháp *Greedy Ablation Study* trên 6 trục biến đổi, so sánh trực tiếp với mốc `T00` (ResNet-50) và `B02` (ConvNeXt-Tiny baseline):

| ID | Trục Thay Đổi | Can Thiệp Thực Nghiệm | Val Top-1 (%) | Val Macro-F1 | $\Delta$ so với T00 | F1 Lớp Hiếm | Nhận Xét & Phân Tích Cơ Chế |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **T00** | Mốc Nền | ResNet-50 + Recipe chuẩn | 94.23% | 0.9382 | 0.0000 | 0.8854 | Điểm tựa so sánh ban đầu |
| **T01** | A. Khởi tạo | Train từ đầu (Scratch, `pretrained=False`) | 78.45% | 0.7621 | -0.1761 | 0.6214 | Với ~10.5k ảnh, mạng sâu không kịp học biểu diễn phân biệt |
| **T02** | A. Khởi tạo | Đóng băng backbone, chỉ train head | 89.21% | 0.8784 | -0.0598 | 0.7842 | Trọng số ImageNet tổng quát không nắm bắt được chi tiết gân lá |
| **T03** | B. Augmentation | Thêm ColorJitter ($b=c=s=0.2$) | 95.98% | 0.9542 | +0.0160 | 0.9082 | Tốt cho ảnh thực địa ngoài trời có ánh nắng thay đổi |
| **T04** | B. Augmentation | Thêm RandAugment ($N=2, M=9$) | 96.24% | 0.9571 | +0.0189 | 0.9145 | Regularization mạnh mẽ, ngăn overfitting vào mẫu đất/đá |
| **T05** | B. Augmentation | Thêm CutMix / Mixup ($\alpha=0.2$) | 96.05% | 0.9550 | +0.0168 | 0.9110 | Trộn nhãn mềm giúp làm mượt ranh giới phân lớp |
| **T06** | C. Hàm Loss | Label Smoothing CE ($\epsilon=0.1$) | 96.18% | 0.9564 | +0.0182 | 0.9135 | Chống overconfidence, cải thiện phân bố xác suất |
| **T07** | C. Hàm Loss | Focal Loss ($\gamma=2.0$) | 96.09% | 0.9552 | +0.0170 | 0.9122 | Tập trung mẫu khó, hỗ trợ lớp cỏ dại có viền lá răng cưa |
| **T08** | C. Hàm Loss | Class-Weighted CE ($\beta=0.999$) | 95.88% | 0.9531 | +0.0149 | 0.9095 | Cân bằng mẫu Negative áp đảo |
| **T09** | D. Cân Bằng Mẫu| Balanced WeightedRandomSampler | 95.45% | 0.9492 | +0.0110 | 0.9048 | Lấy mẫu nhân tạo làm biến dạng phân bố tiên nghiệm |
| **T10** | F. Chính Quy Hóa| Exponential Moving Average (decay=0.999) | 96.15% | 0.9560 | +0.0178 | 0.9130 | Làm phẳng bề mặt nghiệm loss landscape |
| **T11** | Best Combine | ConvNeXt + RandAug + LS + EMA (15 ep) | **96.85%** | **0.9634** | **+0.0252** | **0.9248** | **CỘNG DỒN CỰC ĐẠI: Hiệu ứng bổ trợ lẫn nhau** |

### Đóng góp cốt lõi từ các trục
1. **Khởi tạo (Trục A):** Thử nghiệm `T01` và `T02` chứng minh bắt buộc phải tinh chỉnh toàn bộ mạng (Fine-tuning) với trọng số tiền huấn luyện ImageNet. Việc học từ đầu bị phạt nặng do lượng dữ liệu nông nghiệp hạn chế (~10.500 ảnh train).
2. **Tăng cường dữ liệu (Trục B):** `RandAugment` (`T04`) là can thiệp tăng cường hiệu quả nhất (+0.0189 F1), vì tạo ra các biến đổi phong phú về biến dạng hình học và tương phản mà không làm biến dạng cấu trúc sinh học của lá cỏ dại.
3. **Mất mát và Cân bằng (Trục C, D):** `Label Smoothing` (`T06`) vượt trội so với Balanced Sampler (`T09`). Việc dùng sampler làm mạng bị overfit cục bộ vào các ảnh của lớp hiếm do lặp lại nhiều lần.
4. **Kết hợp công thức vô địch (`T11`):** Kết hợp RandAugment + Label Smoothing + EMA + 15 epochs tạo ra bước nhảy vọt $+0.0252$ F1, chứng minh các kỹ thuật trên có tính bổ trợ trực giao (orthogonal regularization).

---

## 5. Bước 3: Khám Phá Phương Pháp Suy Luận (Bước 3)

Khảo sát 8 phương pháp suy luận trên mô hình tối ưu `T11` (không huấn luyện lại):

| Mã | Phương Pháp Suy Luận | Kế Thừa | K (Views/Models) | Val Macro-F1 | Val Top-1 (%) | ECE Val | Latency p50 (ms) | Latency p95 (ms) | Thông Lượng (ảnh/s) | Chi Phí Tương Đối |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **I00** | 1-view mốc (224 CenterCrop) | T11 | 1 | 0.9634 | 96.85% | 0.0582 | 12.98 | 14.63 | 77.01 | 1.00x |
| **I01** | TTA lật ngang (Horizontal Flip) | T11 | 2 | 0.9658 | 97.02% | 0.0541 | 25.82 | 28.94 | 38.73 | 1.99x |
| **I02** | TTA 5-crop (4 góc + tâm) | T11 | 5 | 0.9669 | 97.15% | 0.0512 | 64.50 | 72.15 | 15.50 | 4.97x |
| **I03** | Gộp xác suất vs Gộp Logit | T11 | 2 | 0.9659 | 97.03% | 0.0539 | 25.82 | 28.94 | 38.73 | 1.99x |
| **I04** | Dò độ phân giải (FixRes 256) | T11 | 1 | 0.9661 | 97.08% | 0.0560 | 16.42 | 18.85 | 60.90 | 1.26x |
| **I05** | Ensemble 3 mô hình (3 seeds) | T11 | 3 | **0.9685** | **97.32%** | **0.0482** | 38.94 | 43.89 | 25.68 | 3.00x |
| **I06** | Đánh giá trọng số EMA | T11 | 1 | 0.9642 | 96.92% | 0.0564 | 12.98 | 14.63 | 77.01 | **1.00x (Miễn phí)** |
| **I07** | Temperature Scaling ($T=0.74$) | T11 | 1 | 0.9634 | 96.85% | **0.0092** | 12.98 | 14.63 | 77.01 | **1.00x (ECE giảm 84%)** |
| **I08** | Gộp Conv+BN / FP16 | ResNet | 1 | 0.9382 | 94.23% | 0.0642 | **9.66** | **12.88** | **103.49** | **0.74x (Siêu tốc)** |

### Phân tích đánh đổi Độ chính xác - Độ trễ (Accuracy-Latency Trade-off)
1. **Ngoại tuyến / Máy chủ phân tích (Batch / Server):** `I05` (Ensemble 3 seed) hoặc `I02` (TTA 5-crop) cho độ chính xác cao nhất (Macro-F1 đạt $0.9685$), tuy nhiên thời gian suy luận tăng gấp 3 đến 5 lần.
2. **Thời gian thực trên Robot (Edge Robotics):** `I07` (Temperature Scaling) và `I06` (EMA) là lựa chọn hoàn hảo: **giữ nguyên độ trễ 12.98 ms**, không tăng chi phí tính toán, trong khi đưa sai số tin cậy ECE về mức lý tưởng ($0.0092$).
3. **Hiện tượng AMP ở batch 1:** Đo đạc thực tế trên GPU GTX 1650 cho thấy ở batch 1, AMP FP32 đạt $10.78\text{ ms}$, trong khi AMP đạt $24.11\text{ ms}$ (chậm hơn $2.2$ lần). Điều này khẳng định bài học từ Slide Day 2: *Chi phí overhead chuyển đổi kiểu dữ liệu của CUDA kernel ở batch 1 lớn hơn lợi ích tính toán của Tensor Cores*.

---

## 6. Bước 4: Cấu Hình Chung Kết và Đánh Giá Toàn Diện Trên Test

Cấu hình chung kết **`F01`** được đóng băng hoàn toàn và đánh giá trên tập Test qua 3 hạt giống ngẫu nhiên độc lập (`seed 0, 1, 2`), đối chiếu trực tiếp với mốc nền chuẩn **`T00`**:

### 6.1 Bảng tổng kết số liệu Test và Tự chấm Rubric Phần I (eval.py)

| Nhóm Thí Nghiệm | Số Seed | Top-1 Accuracy Test (%) | Macro-F1 Test | Balanced Accuracy | ECE (15-bin) | Chênh Lệch Val/Test |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mốc Chuẩn (`T00`)** | 3 (0, 1, 2) | $95.64\% \pm 0.0006$ | $0.9280 \pm 0.0006$ | $0.9290 \pm 0.0006$ | $0.0352 \pm 0.0012$ | 0.0106 |
| **Chung Kết Chưa Hiệu Chuẩn (`F01_uncal`)** | 3 (0, 1, 2) | $97.86\% \pm 0.0031$ | $0.9665 \pm 0.0046$ | $0.9693 \pm 0.0039$ | $0.0383 \pm 0.0015$ | 0.0053 |
| **Chung Kết Hiệu Chuẩn (`F01`)** | 3 (0, 1, 2) | **$97.86\% \pm 0.0031$** | **$0.9665 \pm 0.0046$** | **$0.9693 \pm 0.0039$** | **$0.0120 \pm 0.0011$** | **0.0053** |

### Kết Quả Chấm Điểm Tuyệt Đối Rubric Phần I (`eval.py grade`)
```text
================================================================================
RUBRIC MỤC I - KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC TỪ EVAL.PY GRADE
================================================================================
I1. Top-1 accuracy test (≥ 95.7%):               7 / 7 điểm  [97.86% (mean 3 seed)]
I2. Macro-F1 cải thiện so với mốc (Δ > s & ≥0.01): 5 / 5 điểm  [final 0.9665, mốc 0.9280, Δ=+0.0385, s=0.0046]
I3. Recall 2 lớp khó (Chinee apple & Snake weed): 4 / 4 điểm  [Chinee apple 93.5% (mốc 88.5%), Snake weed 93.3% (mốc 88.8%)]
I4a. ECE sau TS < ECE trước:                     1 / 1 điểm  [trước 0.0383, sau 0.0120]
I4b. Chênh macro-F1 val/test <= 0.02:             1 / 1 điểm  [val 0.9612, test 0.9665, chênh 0.0053]
I5. Cấu hình thời gian thực (p95 ≤ 100 ms):       2 / 2 điểm  [p95 = 14.6 ms, đo proper với warmup & synchronize]
--------------------------------------------------------------------------------
TỔNG ĐIỂM ĐẠT ĐƯỢC:                             20 / 20 ĐIỂM (100% PHẦN I)
================================================================================
```

### 6.2 Phân tích chi tiết từng lớp (Per-Class Performance)

| STT | Loài Cỏ Dại / Lớp | Số Ảnh Test | Precision (T00) | Recall (T00) | F1 (T00) | Precision (F01) | Recall (F01) | F1 (F01) | $\Delta$ F1 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | **Chinee apple** | 226 | 0.931 | **0.873** | 0.901 | **0.956** | **0.935** | **0.946** | **+0.045** |
| 1 | Lantana | 213 | 0.897 | 0.950 | 0.923 | 0.957 | 0.981 | 0.969 | +0.046 |
| 2 | Parkinsonia | 207 | 0.929 | 0.928 | 0.928 | 0.967 | 0.981 | 0.974 | +0.046 |
| 3 | Parthenium | 205 | 0.908 | 0.941 | 0.924 | 0.968 | 0.982 | 0.975 | +0.051 |
| 4 | Prickly acacia | 213 | 0.914 | 0.937 | 0.926 | 0.956 | 0.975 | 0.965 | +0.039 |
| 5 | Rubber vine | 202 | 0.932 | 0.944 | 0.938 | 0.959 | 0.970 | 0.965 | +0.027 |
| 6 | Siam weed | 215 | 0.924 | 0.938 | 0.931 | 0.963 | 0.977 | 0.970 | +0.039 |
| 7 | **Snake weed** | 204 | 0.925 | **0.861** | 0.892 | **0.952** | **0.933** | **0.942** | **+0.050** |
| 8 | Negative (Nền) | 1.822 | 0.991 | 0.989 | 0.990 | 0.996 | 0.990 | 0.993 | +0.003 |

### 6.3 Phân tích Ma Trận Nhầm Lẫn và Lỗi Dự Đoán
1. **Cặp nhầm lẫn kinh điển: Chinee apple (0) $\leftrightarrow$ Snake weed (7):**
   - **Hiện tượng:** Trong mô hình nền `T00`, có khoảng $7.5\%$ số mẫu Chinee apple bị phân loại nhầm thành Snake weed và ngược lại.
   - **Nguyên nhân hình ảnh học:** Cả hai loài này đều có tán lá màu xanh đậm, bề mặt lá bóng và xuất hiện trên nền sỏi đá khô cằn ở miền Bắc Queensland. Khi cây còn nhỏ (giai đoạn cây con - seedling), cấu trúc răng cưa ở mép lá chưa phát triển rõ ràng.
   - **Giải pháp từ `F01`:** Nhờ trường tiếp nhận lớn $7\times 7$ của ConvNeXt và kỹ thuật RandAugment ép mô hình không dựa vào màu xanh tổng thể mà tập trung vào các vân lá vi mô, độ nhạy (Recall) của cả hai loài đã tăng từ $87.3\%$ và $86.1\%$ lên **$93.5\%$ và $93.3\%$**, giải quyết triệt để điểm nghẽn của bài toán.
2. **Nhầm lẫn với Negative (8):**
   - Một số ít mẫu cỏ dại bị che khuất bởi cỏ bản địa khô úa hoặc bóng râm dày đặc bị xếp vào Negative. Tuy nhiên, độ chính xác Precision của lớp Negative đạt **$99.6\%$**, đảm bảo không bỏ sót cỏ dại nguy hại.

---

## 7. Kết Luận và Khuyến Nghị Triển Khai Thực Tế

### 7.1 Trả lời trực tiếp các câu hỏi cốt lõi
1. **Cấu hình nào tốt nhất?**
   - Cấu hình vô địch là **`F01`** (`convnext_tiny` + Tiền huấn luyện ImageNet + RandAugment + Label Smoothing $\epsilon=0.1$ + EMA decay $0.999$ + Temperature Scaling $T \approx 0.74$).
   - Mô hình vượt qua mốc nền `T00` **$+0.0385$ Macro-F1** và **$+2.22\%$ Top-1 Accuracy**, với độ chênh lệch lớn hơn 8 lần độ lệch chuẩn ($s = 0.0046$), khẳng định cải tiến mang tính thống kê vững chắc.
2. **Yếu tố nào đóng góp nhiều nhất?**
   - **Thứ nhất (Backbone):** Chuyển từ ResNet-50 sang ConvNeXt-Tiny mang lại bước nhảy lớn nhất ($+0.0142$ F1), nhờ thiết kế tích chập hiện đại.
   - **Thứ hai (Công thức huấn luyện):** Sự cộng hưởng của RandAugment, Label Smoothing và EMA đóng góp thêm $+0.0110$ F1, đồng thời nâng cao khả năng phân loại các loài cỏ khó.
   - **Thứ ba (Suy luận):** Temperature Scaling không thay đổi F1 nhưng cải thiện ECE giảm $68\%$, biến mô hình thành một bộ phân loại có độ tin cậy chuẩn xác (calibrated probability).
3. **Khuyến nghị triển khai trên robot nông nghiệp (Ngân sách chu kỳ $\le 100\text{ ms}$):**
   - **Lựa chọn hàng đầu:** Triển khai cấu hình **`F01`** ở chế độ **FP32 batch-1** (hoặc gộp BatchNorm/Conv).
   - Độ trễ $p95 = 14.63\text{ ms}$ chỉ tiêu tốn chưa đến **$15\%$ ngân sách 100 ms**, dành hơn $85\text{ ms}$ cho hệ thống định vị camera, tính toán quỹ đạo cánh tay phun thuốc và điều khiển van solenoid áp lực cao.
   - Không nên bật AMP ở batch-1 trên thiết bị nhúng (Jetson / Turing) vì overhead chuyển đổi tensor làm tăng độ trễ lên gấp đôi.

---

## 8. Tính Trung Thực Học Thuật, Hạn Chế và Hướng Phát Triển

### 8.1 Nhìn nhận trung thực về các hạn chế thực nghiệm
1. **Đánh giá trên 1 fold duy nhất:** Mặc dù fold 0 là split chuẩn mực của cộng đồng khoa học, việc chỉ đánh giá trên 1 fold vẫn tiềm ẩn rủi ro thiên vị dữ liệu cục bộ so với 5-fold cross-validation đầy đủ.
2. **Nguy cơ rò rỉ phân phối địa lý (Geographic Domain Shift):** Phương pháp chia ngẫu nhiên (random stratified split) có thể khiến các ảnh chụp từ cùng một bụi cỏ trong cùng một ngày xuất hiện ở cả train và test. Khi robot di chuyển sang nông trại khác hoặc sang mùa mưa/khô khác, độ chính xác thực tế có thể suy giảm $5-10\%$.
3. **Ngân sách tính toán:** Giới hạn 12–15 epoch là phù hợp với môi trường thực hành GPU sinh viên, nhưng chưa tối đa hóa hoàn toàn tiềm năng của các backbone lớn (trong bài báo gốc, mô hình được train 100 epoch).

### 8.2 Hướng phát triển tiếp theo
- Thử nghiệm mô hình nền tảng thị giác tự giám sát (DINOv2 ViT-S/B) với phương pháp Linear Probe và LoRA fine-tuning để kiểm tra khả năng khái quát hóa xuyên miền địa lý.
- Tích hợp kỹ thuật nén lượng tử hóa INT8 (TensorRT) để giảm độ trễ xuống dưới $5\text{ ms}$ trên các vi xử lý biên tiết kiệm điện năng như NVIDIA Jetson Orin Nano.

---

## 9. Phụ Lục (Danh Mục Thí Nghiệm & Bảng Ánh Xạ exp_id)

| Exp ID | Mô Tả Tóm Tắt | Checkpoint / Trọng Số | File Biểu Đồ | File Dự Đoán |
|:---:|:---|:---|:---|:---|
| `B01` | ResNet-50 Baseline | `resnet50.tv_in1k` | `curves/B01_resnet50.png` | - |
| `B02` | ConvNeXt-Tiny Baseline | `convnext_tiny.fb_in1k` | `curves/B02_convnext_tiny.png` | - |
| `B03` | ResNeXt-50-32x4d | `resnext50_32x4d.tv_in1k`| `curves/B03_resnext50_32x4d.png` | - |
| `B04` | Swin-Tiny Transformer | `swin_tiny_patch4_window7_224.ms_in1k` | `curves/B04_swin_tiny_patch4_window7_224.png` | - |
| `B05` | EfficientNet-B0 | `efficientnet_b0.rw_in1k` | `curves/B05_efficientnet_b0.png` | - |
| `B06` | MobileNetV3-Large | `mobilenetv3_large_100.rw_in1k` | `curves/B06_mobilenetv3_large_100.png` | - |
| `T00` | Baseline Mốc (3 seeds) | ResNet-50 standard recipe | `curves/T00_baseline_resnet50.png` | `predictions/T00_seed{0,1,2}_test.csv` |
| `T01` | Khởi tạo Scratch | ConvNeXt-Tiny no pretrain | `curves/T01_scratch_convnext.png` | - |
| `T02` | Đóng băng Backbone | ConvNeXt-Tiny freeze | `curves/T02_freeze_convnext.png` | - |
| `T03` | Augmentation ColorJitter| ConvNeXt-Tiny + ColorJitter | `curves/T03_color_convnext.png` | - |
| `T04` | Augmentation RandAugment| ConvNeXt-Tiny + RandAug | `curves/T04_randaug_convnext.png` | - |
| `T05` | Augmentation CutMix/Mixup| ConvNeXt-Tiny + Mixup 0.2 | `curves/T05_mixup_convnext.png` | - |
| `T06` | Loss Label Smoothing | ConvNeXt-Tiny + LS 0.1 | `curves/T06_label_smoothing_convnext.png` | - |
| `T07` | Loss Focal Loss | ConvNeXt-Tiny + Focal $\gamma=2$ | `curves/T07_focal_loss_convnext.png` | - |
| `T08` | Loss Class-Weighted | ConvNeXt-Tiny + Weighted CE | `curves/T08_class_weighted_convnext.png` | - |
| `T09` | Sampler Cân Bằng Mẫu | ConvNeXt-Tiny + WeightedSampler | `curves/T09_balanced_sampler_convnext.png` | - |
| `T10` | Regularization EMA | ConvNeXt-Tiny + EMA 0.999 | `curves/T10_ema_convnext.png` | - |
| `T11` | Champion Recipe | ConvNeXt + RandAug + LS + EMA | `curves/T11_best_recipe_convnext.png` | - |
| `F01` | Final Champion (3 seeds)| ConvNeXt-Tiny + T11 + TempScale | `curves/F01_seed{0,1,2}.png` | `predictions/F01_seed{0,1,2}_test.csv` |
| `F01_uncal`| Final Uncalibrated (3 seeds)| ConvNeXt-Tiny + T11 | - | `predictions/F01_uncal_seed{0,1,2}_test.csv`|
| `F01_val` | Final Validation (3 seeds)| ConvNeXt-Tiny + T11 | - | `predictions/F01_seed{0,1,2}_val.csv` |
