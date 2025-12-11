# src/training.py
import pandas as pd
import logging
import json
from typing import Tuple
from typing import Dict, Any, Union, Tuple
from sklearn.base import BaseEstimator
import numpy as np
from src.utils.tuner import HyperparamTuner 
from src.utils.result_manager import ModelEvaluator, ModelPersister




logger = logging.getLogger('TrainingPipeline')

DataArray = Union[np.ndarray, pd.DataFrame]
TargetArray = Union[np.ndarray, pd.Series]

class TrainingPipeline:
    '''
    TrainingPipeline: Điều phối toàn bộ quy trình huấn luyện,
    tận dụng HyperparamTuner, ModelEvaluator, và ModelPersister.
    '''
    
    SCORING_METRICS = {
        "neg_mean_squared_error": "neg_mean_squared_error",
        "r2": "r2"
    }
    REFIT_METRIC = "neg_mean_squared_error"

    def __init__(self, X_train: DataArray, 
                 y_train: TargetArray, 
                 X_test: DataArray, 
                 y_test: TargetArray,
                 random_seed: int,
                run_config: Dict[str, Any]):
        
        # Thuộc tính dữ liệu
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Thuộc tính cấu hình chạy
        self.seed = random_seed
        self.run_config = run_config # Chứa model_name, paths, run_id, tuner_flag,...
        self.is_tuned = bool(run_config['tuner'])

        # 3. Thuộc tính trạng thái
        self.best_model: BaseEstimator = None
        self.best_params: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}

    def _load_param_grid(self, config_path: str, model_name: str) -> Dict[str, Any]:
        """Load param grid theo model từ hyperarams.json."""
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            return cfg.get(model_name, {})
        except Exception as e:
            logger.error(f"Lỗi khi đọc file hyperparams.json: {e}")
            return {}

    def run_pipeline(self) -> Tuple[BaseEstimator, Dict[str, float]]:
        """Điều phối việc train, tune, đánh giá và lưu trữ."""

        model_name = self.run_config['model_name']
        config_path = self.run_config['hyperparam_config_path']
        
        # ---Chuẩn bị Tuning/Training ---
        param_grid = {}
        if self.is_tuned:
            logger.info("Chế độ: HYPERPARAMETER TUNING đang bật.")
            param_grid = self._load_param_grid(config_path, model_name)
        else:
            logger.info("Chế độ: BASE TRAINING (Không tuning).")
        
        # ---  Huấn luyện (tuning) ---
        tuner = HyperparamTuner(
            model_name=model_name,
            scoring_metrics=self.SCORING_METRICS,
            refit_metric=self.REFIT_METRIC,
            param_grid=param_grid,
            gpu =  self.run_config['gpu'],
            seed=self.seed
        )
        self.best_model, self.best_params = tuner.run_grid_search(self.X_train, self.y_train)
        
        # ---  Đánh giá (Evaluate) ---
        evaluator = ModelEvaluator(self.X_test, self.y_test)
        self.metrics = evaluator.evaluate(self.best_model)
        
        # Visualizations
        visualize_path = evaluator.visualize_predictions(
            model_name=model_name,
            is_tuned=self.is_tuned
        )
        logger.info(f"→ Saved visualization: {visualize_path}")
        
        # Residuals Plot
        visualize_residuals_path = evaluator.visualize_residuals(
            model_name=model_name,
            is_tuned=self.is_tuned
        )
        logger.info(f"→ Saved residuals plot: {visualize_residuals_path}")

        # Feature Importance (nếu chạy XGBRegressor)
        visualize_path_fi = evaluator.visualize_feature_importance(
            model=self.best_model,
            model_name=model_name,
            is_tuned=self.is_tuned,
            top_n=15 # Chỉ hiển thị Top 15 feature (có thể tùy chỉnh)
        )
        if visualize_path_fi:
            logger.info(f"→ Saved Feature Importance: {visualize_path_fi}")
        # Lưu trữ (Persist) 
        persister = ModelPersister(
            run_id=self.run_config['run_id'],
            model_name=model_name,
            is_tuned=self.is_tuned,
            random_seed=self.seed
        )
        
        #  Lưu Model
        model_save_path = persister.save_model(
            self.best_model, 
            save_dir=self.run_config['savedir_models']
        )
        logger.info(f"→ Saved model to : {model_save_path}")

        #  Lưu Metrics
        persister.save_results(
            metrics=self.metrics,
            best_params=self.best_params,
            filepath="outputs/metrics/experiment_log.csv"
        )
        
        logger.info("\n===== PIPELINE COMPLETED SUCCESSFULLY =====\n")
        return self.best_model, self.metrics

class DataIOLoader:
    """
    Class chịu trách nhiệm Nạp dữ liệu (I/O) từ các file đã xử lý.
    """

    @staticmethod
    def load_processed_data(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load dữ liệu X_train, X_test, y_train, y_test từ các đường dẫn trong args."""
        try:
            logger.info("Đang nạp dữ liệu đã xử lý...")
            X_train = pd.read_csv(args.X_train_path)
            X_test = pd.read_csv(args.X_test_path)
            # Sử dụng .squeeze() để đảm bảo y là Series/numpy array 1D
            y_train = pd.read_csv(args.y_train_path).squeeze()
            y_test = pd.read_csv(args.y_test_path).squeeze()
            logger.info("Nạp dữ liệu hoàn tất.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Lỗi khi nạp dữ liệu từ các file: {e}")
            raise