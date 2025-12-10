# src/results_manager.py
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor 
from typing import Dict, Any, Union, Tuple
import seaborn as sns
from src.utils.metrics import RegressionMetrics 

# --- LOGGER SETUP ---
logger = logging.getLogger('ResultsManager') # Sử dụng logger chung cho file


# CLASS 1: ModelEvaluator

class ModelEvaluator:
    """
    Class chịu trách nhiệm Đánh giá mô hình đã huấn luyện (Prediction & Metrics).
    """

    def __init__(self, X_test: pd.DataFrame, y_test: pd.Series):
        self.X_test = X_test
        self.y_test = y_test
        self.metrics: Dict[str, float] = {}
        self.y_pred: np.ndarray = None

    def evaluate(self, model: BaseEstimator) -> Dict[str, float]:
        """
        Dự đoán trên tập Test và tính toán các metrics.
        """
        logger.info("Bắt đầu đánh giá mô hình trên tập Test...")
        
        # Dự đoán
        self.y_pred = model.predict(self.X_test)

        # Tính metrics
        self.metrics = RegressionMetrics.compute_metrics(self.y_test, self.y_pred)
        
        # Logging kết quả
        logger.info(f"--- KẾT QUẢ ĐÁNH GIÁ ---")
        for metric, value in self.metrics.items():
             logger.info(f"{metric}: {value:.4f}")
        logger.info("--------------------------")

        return self.metrics

    def visualize_predictions(self, model_name: str, is_tuned: bool, sample_size: int = 200) -> str:
        """
        Vẽ biểu đồ đường Y_true vs Y_pred trên mẫu nhỏ.
        """
        if self.y_pred is None:
            logger.warning("Chưa chạy evaluate. Không thể visualize.")
            return ""

        y_true = self.y_test.values
        
        # Lấy sample nhỏ
        n = min(sample_size, len(y_true))
        y_true_sample = y_true[:n]
        y_pred_sample = self.y_pred[:n]
        x_axis = np.arange(n)

        # Plotting
        plt.figure(figsize=(14, 6))
        plt.plot(x_axis, y_true_sample, label="Y True", linewidth=2)
        plt.plot(x_axis, y_pred_sample, label="Y Pred", linewidth=2, linestyle='--')

        plt.title(f"{model_name} – Line Chart (First {n} Samples)")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save file
        save_dir = "outputs/visualize_images"
        os.makedirs(save_dir, exist_ok=True)
        tune_tag = 1 if is_tuned else 0

        save_path = os.path.join(
            save_dir,
            f"{model_name}_linechart_{n}_samples_{tune_tag}_tuning.png"
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return save_path


    def visualize_residuals(self, model_name: str, is_tuned: bool, bins: int = 50) -> str:
            """
            Vẽ biểu đồ phân phối Residuals (Sai số) dưới dạng Histogram và KDE.
            Mục tiêu: Kiểm tra giả định phân phối chuẩn của sai số.
            """

            if self.y_pred is None:
                logger.warning("Chưa chạy evaluate. Không thể visualize.")
                return ""
            y_true = self.y_test.values
            residuals = y_true - self.y_pred
            
            # Setup Plot
            plt.figure(figsize=(10, 6))
            sns.histplot(
                residuals, 
                bins=bins, 
                kde=True, 
                color="skyblue", 
                edgecolor="black", 
                alpha=0.7,
                stat="density", 
                label="Residuals Histogram"
            )

            # Thêm đường phân phối chuẩn lý thuyết (Gaussian/Normal Distribution)
            mu, std = residuals.mean(), residuals.std()
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mu) ** 2) / (2 * std ** 2))
            
            plt.plot(x, p, 'k', linewidth=2, label="Normal Distribution (Theoretical)")

            title = f"{model_name} – Residuals Distribution (Tuning: {'Yes' if is_tuned else 'No'})"
            plt.title(title)
            plt.xlabel("Residual Value (Y_true - Y_pred)")
            plt.ylabel("Frequency / Density")
            plt.legend()
            plt.grid(axis='y', alpha=0.5)

            # Save file
            save_dir = "outputs/visualize_images"
            os.makedirs(save_dir, exist_ok=True)
            tune_tag = 1 if is_tuned else 0
            
            save_path = os.path.join(
                save_dir,
                f"{model_name}_residuals_{tune_tag}.png"
            )

            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

            return save_path

# === PHƯƠNG THỨC : VISUALIZE FEATURE IMPORTANCE ===
    def visualize_feature_importance(self, model: BaseEstimator, model_name: str, is_tuned: bool, top_n: int = 15) -> str:
        """
        Trích xuất và vẽ biểu đồ Feature Importance ('gain') cho mô hình XGBoost.
        """
        
        # 1. Kiểm tra mô hình
        if not isinstance(model, XGBRegressor):
            logger.warning(f"Mô hình '{model_name}' không phải là XGBRegressor. Không thể tính Feature Importance theo 'gain'.")
            return ""

        logger.info("Bắt đầu tính Feature Importance bằng XGBoost (Type: 'gain')...")

        # 2. Trích xuất Feature Importance
        booster = model.get_booster()
        
        # Lấy Importance theo 'gain' (Độ giảm mất mát trung bình mà feature đóng góp)
        f_importance = booster.get_score(importance_type='gain')
        
        if not f_importance:
             logger.warning("Không có Feature Importance nào được tính toán. (Có thể do mô hình base không được huấn luyện).")
             return ""
        
        # 3. Chuẩn bị dữ liệu và sắp xếp
        df_importance = pd.DataFrame(
            list(f_importance.items()), 
            columns=['Feature', 'Importance']
        ).sort_values(by='Importance', ascending=False)

        # Lấy Top N
        df_top_n = df_importance.head(top_n)

        # 4. Vẽ biểu đồ
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=df_top_n, 
            palette='viridis'
        )

        title = f"{model_name} – Top {top_n} Feature Importance (Gain) (Tuning: {'Yes' if is_tuned else 'No'})"
        plt.title(title)
        plt.xlabel("Feature Importance ('gain')")
        plt.ylabel("Features")
        plt.tight_layout()

        # 5. Lưu file
        save_dir = "outputs/visualize_images"
        os.makedirs(save_dir, exist_ok=True)
        tune_tag = 1 if is_tuned else 0
        
        save_path = os.path.join(
            save_dir,
            f"{model_name}_feature_importance_{tune_tag}.png"
        )

        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Đã lưu biểu đồ Feature Importance tại: {save_path}")
        return save_path
# ============================================================================
# CLASS 2: ModelPersister

class ModelPersister:
    """
    Class chịu trách nhiệm Lưu trữ: model, metrics log.
    """
    
    @staticmethod
    def timestamp() -> str:
        """Trả về timestamp thống nhất."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __init__(self, run_id: str, model_name: str, is_tuned: bool, random_seed: int):
        self.run_id = run_id
        self.model_name = model_name
        self.is_tuned = is_tuned
        self.random_seed = random_seed
        self.results_dict: Dict[str, Any] = {}

    def save_model(self, model: BaseEstimator, save_dir: str) -> str:
        """Lưu mô hình đã huấn luyện vào file .pkl."""
        
        tune_tag = 1 if self.is_tuned else 0
        filename = f"{self.run_id}_{self.model_name}_{tune_tag}_tuner.pkl"
        filepath = os.path.join(save_dir, filename)

        try:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(model, f)
            
            logger.info(f"Đã lưu mô hình .pkl tại: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình .pkl: {e}")
            return ""

    def save_results(self, metrics: Dict[str, float], best_params: Dict[str, Any], filepath: str):
        """Lưu kết quả đánh giá (metrics & params) vào file CSV."""
        
        self.results_dict = {
            'timestamp': self.timestamp(),
            'tuning': 'Yes' if self.is_tuned else 'No',
            'model': self.model_name,
            'random_seed': self.random_seed,
            **metrics, 
            'best_params': str(best_params) # Lưu params dưới dạng chuỗi
        }
        
        try:
            results_df = pd.DataFrame([self.results_dict])
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            
            # Ghi vào file (append nếu đã tồn tại)
            if os.path.exists(filepath):
                 results_df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                 results_df.to_csv(filepath, mode='w', header=True, index=False)
                 
            logger.info(f"Đã lưu kết quả thí nghiệm vào {filepath}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu kết quả: {e}")