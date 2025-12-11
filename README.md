# Dự đoán Giá Xe Cũ bằng Machine Learning  
Đồ án Cuối kỳ – Machine Learning / Data Science


## Nhóm 14 
| Họ tên | MSSV |
|--------|------|
| Nguyễn Thị Huyền Trang | 23280092 |
| Đặng Trọng Bảo Thi     |23280025 |
| Lưu Vũ Lâm            | 23280067 |
---
## 1. Giới thiệu Dự án

**Mô tả:**  
Dự án xây dựng một pipeline Khoa học Dữ liệu hoàn chỉnh để dự đoán giá xe cũ dựa trên các đặc trưng như hãng xe, năm sản xuất, số km đã chạy, dung tích động cơ, loại hộp số,…

Pipeline bao gồm:
- Tiền xử lý dữ liệu (cleaning, encoding, scaling,…)
- Phân tích khám phá dữ liệu (EDA)
- Huấn luyện nhiều thuật toán ML (Linear Regression, XGBoost, LGBM)
- Tối ưu siêu tham số (Hyperparameter Tuning)
- Trực quan phân tích & Đánh giá & so sánh mô hình
- Lưu mô hình và log toàn bộ quá trình

**Dataset sử dụng:**  
[DATASET](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/data?fbclid=IwY2xjawOnrwNleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeQwBbAj4Poh5uexVolTpguD-KCdhvh0KPu_kqH4lUDqcywzBPTyGtVgsX7Fc_aem_M8xdDIrOTkasNR49_Aim6w)

---




# 2. Cấu trúc Thư mục
```

.
├── config/
│   └── hyperparams.json            # Cấu hình siêu tham số.
│
├── notebooks/
│   └── EDA_explore_data.ipynb      # Notebook phân tích dữ liệu.
│
├── outputs/
│   ├── logs/                       # File log huấn luyện và tiền xử lý.
│   ├── metrics/                    # Kết quả đánh giá mô hình (json/csv).
│   ├── models/                     # Lưu mô hình đã train (.pkl).
│   └── visualize_images/           # Lưu biểu đồ so sánh/tương quan 
│
├── output_experiment/              # Kết quả thí nghiệm của nhóm.
│
├── run_scripts/
│   ├── run_data_process.sh         # Script chạy xử lý dữ liệu.
│   └── run_training.sh             # Script chạy huấn luyện mô hình.
│
├── src/
│   ├── utils/
│   │   ├── data_config.py          # Đường dẫn dữ liệu, cấu hình IO.
│   │   ├── metrics.py              # Hàm tính RMSE, MAE, R2,...
│   │   ├── tuner.py                # Tối ưu siêu tham số (Grid/Random).
│   │   ├── model_factory.py        # Khởi tạo model theo tên.
│   │   ├── result_manager.py       # Lưu kết quả: metrics, hình ảnh,...
│   │
│   │── training.py                 # Class TrainingPipeline (train + eval).
│   │
│   ├── data_processor.py           # Class điều phối DataProcessor
│
├── main_script.py                  # Chạy toàn bộ pipeline end-to-end.
│
├── .gitignore
├── requirements.txt
└── README.md

```



# 3. Cài đặt & Yêu cầu

## Yêu cầu hệ thống
- Python 3.8+
- pip 20+
- RAM ≥ 8GB
- cpu
## Tải dự án 
```bash 
git clone https://github.com/huyentranq/Car-Price-Prediction.git
cd Car-Price-Prediction
```
## Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate      # MacOS/Linux
# venv\Scripts\activate         # Windows
```
Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## Tải Dataset và chuẩn bị thư mục

Data xem tại [DATASET](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/data?fbclid=IwY2xjawOnrwNleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeQwBbAj4Poh5uexVolTpguD-KCdhvh0KPu_kqH4lUDqcywzBPTyGtVgsX7Fc_aem_M8xdDIrOTkasNR49_Aim6w)


**Tạo thư mục data/ trong project**

Đặt file CSV vào:
``` bash
.
└── data/
    └── vehicles.csv        
```



# 4. Hướng dẫn chạy
## 4.1. Chạy tiền xử lý dữ liệu
```bash
bash run_scripts/data_processing.sh

# python3 main_script.py \
#     --data_process 1\   # bật tiền xử lý dữ liệu 
#     --data_input_path "data/vehicles.csv" \   # data đầu vào
#     --data_output_path "Data_processed" \     # folder lưu kết quả
```


**Output**: folder **Data_processed** chứa data đã qua xử lý
## 4.2. Train mô hình
**Chạy lần lượt từng thí nghiệm sau (không chạy đồng thời)**
```bash
# Train Linear Regression (Có Tuning & không Tuning):
bash run_scripts/run_model_LinearRegression.sh
```
```bash
# Train LGBM (Có Tuning & không Tuning):
bash run_scripts/run_model_LGBMRegressor.sh
```
```bash
# Train XGBRegressor (Có Tuning & không Tuning)
bash run_scripts/run_model_XGBRegressor.sh
```

### Script CLI tùy chỉnh
``run_scripts/`` cung cấp **các script CLI tùy chỉnh** để bạn linh hoạt chạy thử nhiều mô hình.

Ví dụ dưới đây là **kịch bản chạy XGBRegressor không tuning**

```bash
# --- KỊCH BẢN: XGBRegressor - Không Tuning ---
# Có thể tùy chỉnh tham số để chạy nhiều lần thử nghiệm:
#   --model_name       : chọn model muốn chạy (LinearRegression, LGBMRegressor, XGBRegressor)
#   --tuner            : 0 = không tuning, 1 = bật tuning
#   --run_id           : gán ID cho lần chạy (dùng để lưu file không bị trùng)
#   --savedir_models   : nơi lưu model
#
# Ví dụ này: chạy XGBRegressor không tuning, run_id = 005

echo -e "\n--- [C] KỊCH BẢN: XGBRegressor - Không TUNING ---"
python main_script.py \
    --model_name "XGBRegressor" \
    --tuner 0 \
    --run_id "005" \
    --savedir_models "outputs/models"


→ Mỗi lần chạy sẽ sinh ra một bộ logs + metrics + model riêng, không bị ghi đè.

```


### Feature Importance:
---
Mạc định train XGBRegressor, kiểm tra kết quả tại: **outputs/visualize_images/**

```bash 
python main_script.py \
    --model_name "XGBRegressor" \
    --tuner 0 \
    --run_id "005" \
    --savedir_models "outputs/models"
```

### **Output của training** :(tham khảo) [output_experiment](https://github.com/huyentranq/Car-Price-Prediction-ML/tree/master/output_experiment)
```bash
├── outputs/
│   ├── logs/                   # Log quá trình huấn luyện/xử lý.
│   ├── metrics/                # best_parameter & Chỉ số đánh giá (RMSE, MAE, R2,...). 
│   ├── models/                 # Mô hình đã train (.pickle).
|   └── visualize_images/       # Ảnh trực quan giữa y_true & y_predict
```
---





---
# 5. Kết quả mô hình

## 5.1. So sánh các mô hình & Tuning

| Tuning | Model             | MAE     | MSE           | RMSE    | R2     |
|--------|-------------------|---------|---------------|---------|--------|
| No     | LinearRegression  | 4751.22 | 55940288.29   | 7479.32 | 0.7265 |
| No     | XGBRegressor      | 3075.71 | 27818324.00   | 5274.30 | 0.8640 |
| Yes    | XGBRegressor      | 2609.78 | 21328510.00   | 4618.27 | 0.8957 |
| No     | LGBMRegressor     | 3384.21 | 32119118.45   | 5667.37 | 0.8429 |
| Yes    | LGBMRegressor     | 2873.44 | 24123448.41   | 4911.56 | 0.8820 |

### Nhận xét nhanh
- **XGBRegressor + Tuning cho kết quả tốt nhất (R2 ≈ 0.896)** → mô hình phù hợp nhất.  
- **LGBMRegressor + Tuning** đạt kết quả tốt thứ hai.  
- **Linear Regression** đơn giản, nên sai số cao hơn và không phù hợp dữ liệu phi tuyến.

---

# 6. Những điểm nổi bật của dự án

- Có **pipeline đầy đủ**: preprocessing → training → evaluation.  
- **Tách code theo modules**, dễ bảo trì và mở rộng.  
- **Logging đầy đủ** theo từng bước để theo dõi quá trình huấn luyện.  
- Hỗ trợ **hyperparameter tuning** (GridSearch / Optuna).  
- Lưu **mô hình**, **metrics**, **log** theo từng lần chạy.  
- Có **shell scripts** tự động hóa training và preprocessing.

---

# 7. Hướng phát triển

- Tích hợp **FastAPI** để xây API dự đoán giá xe theo input người dùng.  
- Tự động hóa pipeline với **Airflow hoặc Prefect**.  

- Triển khai mô hình lên **AWS EC2 / SageMaker**, hoặc Render.  
- Xây dựng dashboard trực quan bằng **Streamlit** để demo.

