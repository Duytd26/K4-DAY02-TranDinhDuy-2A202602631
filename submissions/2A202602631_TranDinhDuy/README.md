# BÀI NỘP LAB DAY 2 - DEEPWEEDS (PHÂN LOẠI CỎ DẠI NÂNG CAO)

**Học viên:** Trần Đình Duy  
**MSSV:** 2A202602631  
**Lớp / Khóa:** K4 Deep Learning Advanced  
**Repository GitHub:** [https://github.com/Duytd26/K4-DAY02-TranDinhDuy-2A202602631](https://github.com/Duytd26/K4-DAY02-TranDinhDuy-2A202602631)  
**Notebook Google Colab (Chạy lại trực tiếp):** [lab_day2.ipynb trên Google Colab](https://colab.research.google.com/github/Duytd26/K4-DAY02-TranDinhDuy-2A202602631/blob/main/code/lab_day2.ipynb)

---

## 1. Cấu Trúc Thư Mục Nộp Bài
```text
submissions/2A202602631_TranDinhDuy/
├── README.md              # Hướng dẫn tái lập thực nghiệm và link Colab
├── report.md              # Báo cáo thực nghiệm khoa học 8 phần chi tiết
├── results.xlsx           # Bảng tổng hợp số liệu 7 sheets chuẩn hoá
├── code/                  # Toàn bộ mã nguồn hoàn chỉnh (không còn NotImplementedError)
│   ├── dataset.py         # Xử lý dữ liệu, kiểm định S1-S6, transforms, DataLoader
│   ├── model.py           # Khởi tạo mô hình, backbone timm, đếm tham số & GMACs
│   ├── losses.py          # Label Smoothing, Focal Loss, Class Weights, CutMix/Mixup
│   ├── train.py           # Vòng lặp huấn luyện, Cosine Warmup, AMP, EMA, evaluate
│   ├── inference.py       # TTA, FixRes, Ensembling, Temperature Scaling, Fused BN
│   ├── benchmark.py       # Đo độ trễ chuẩn mực (warmup, cuda.synchronize, p50/p95/p99)
│   └── lab_day2.ipynb     # Notebook tổng hợp từ Bước 0 đến Bước 5
├── curves/                # 21 biểu đồ training curve riêng biệt (B01-B06, T00-T11, F01_seeds)
└── predictions/           # 12 file CSV dự đoán đạt chuẩn (F01 seeds 0,1,2, T00 seeds 0,1,2, uncal, val)
```

---

## 2. Môi Trường Huấn Luyện và Thư Viện
- **Hệ điều hành:** Windows 10 / Ubuntu 22.04 LTS / Google Colab Linux
- **Python:** `3.10` - `3.13`
- **PyTorch:** `2.7.1+cu118` (hỗ trợ CUDA 11.8 / 12.x)
- **Thư viện phụ trợ:**
  - `timm >= 1.0.14`
  - `torchvision >= 0.20.0`
  - `pandas >= 2.0.0`
  - `numpy >= 1.24.0`
  - `openpyxl >= 3.1.0`
  - `matplotlib >= 3.7.0`
- **Phần cứng thực nghiệm:** NVIDIA GeForce GTX 1650 (4GB VRAM) / NVIDIA Tesla T4 (Colab)

---

## 3. Hướng Dẫn Tái Lập Kết Quả (Reproducibility Guide)

### Cách 1: Chạy trực tiếp trên Google Colab
1. Mở notebook qua liên kết: [lab_day2.ipynb trên Google Colab](https://colab.research.google.com/github/Duytd26/K4-DAY02-TranDinhDuy-2A202602631/blob/main/code/lab_day2.ipynb)
2. Chọn Runtime: **Change runtime type** $\to$ **T4 GPU** $\to$ **Save**.
3. Chọn menu **Runtime** $\to$ **Run all** (hoặc nhấn `Ctrl + F9`).
4. Toàn bộ quy trình từ tải dữ liệu, giải nén, kiểm định S1–S6, huấn luyện và đánh giá sẽ tự động thực hiện từ đầu đến cuối.

### Cách 2: Chạy cục bộ qua dòng lệnh (Local CLI)
1. Clone repository và cài đặt thư viện:
   ```bash
   git clone https://github.com/Duytd26/K4-DAY02-TranDinhDuy-2A202602631.git
   cd K4-DAY02-TranDinhDuy-2A202602631
   pip install -r requirements.txt
   ```
2. Kiểm tra bộ unit tests:
   ```bash
   python -m unittest discover -s tests -v
   ```
3. Chấm điểm chính thức bằng công cụ `eval.py`:
   - Tính điểm toàn diện phần I Rubric:
     ```bash
     python eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --uncal "predictions/F01_uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --latency-p95-ms 14.63 --latency-method proper --test-csv "data/labels/test_subset0.csv" --val-csv "data/labels/val_subset0.csv" --labels "data/labels/labels.csv"
     ```
   - Đo chỉ số chi tiết cho cấu hình chung kết `F01`:
     ```bash
     python eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv "data/labels/test_subset0.csv" --labels "data/labels/labels.csv" --tag F01_test
     ```

---

## 4. Tóm Tắt Kết Quả Đạt Được (Điểm Rubric Phần I: 20/20)
- **Top-1 Accuracy Test:** **$97.86\% \pm 0.0031$** (Đạt mức tối đa $\ge 95.7\% \to 7/7$ điểm).
- **Macro-F1 Cải thiện:** **$0.9665 \pm 0.0046$** (Tăng $+0.0385$ so với mốc $0.9280$, vượt $s=0.0046$ và $\ge 0.01 \to 5/5$ điểm).
- **Recall 2 lớp khó:** Chinee apple đạt **$93.5\%$** (mốc $88.5\%$), Snake weed đạt **$93.3\%$** (mốc $88.8\%$) $\to 4/4$ điểm.
- **Hiệu chuẩn tin cậy (I4a):** ECE test giảm từ $0.0383$ xuống **$0.0120$** sau Temperature Scaling $\to 1/1$ điểm.
- **Độ ổn định Val/Test (I4b):** Chênh lệch giữa Val F1 ($0.9612$) và Test F1 ($0.9665$) là **$0.0053 \le 0.02$** $\to 1/1$ điểm.
- **Thời gian thực Robot (I5):** Độ trễ batch-1 $p95 = 14.63\text{ ms} \le 100\text{ ms}$, đo chuẩn với warmup và synchronize $\to 2/2$ điểm.
