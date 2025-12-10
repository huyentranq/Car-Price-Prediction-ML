from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class RegressionMetrics:
    """
    Tính toán các chỉ số đánh giá cho bài toán hồi quy:
    - MAE  (Mean Absolute Error)
    - MSE  (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - R2 Score
    """

    @staticmethod
    def compute_metrics(y_true, y_pred):
        """
        Trả về dict chứa toàn bộ metric.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

    @staticmethod
    def print_metrics(metrics_dict):
        """
        In metric ra màn hình.
        """
        print("\n===== Regression Metrics =====")
        for key, value in metrics_dict.items():
            print(f"{key}: {value:.4f}")
        print("==============================\n")
