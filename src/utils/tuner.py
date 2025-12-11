# src/utils/tuner.py
from sklearn.model_selection import GridSearchCV
import logging
from typing import Tuple, Dict, Any, Union
from sklearn.base import BaseEstimator

from src.utils.model_factory import ModelFactory 

logger = logging.getLogger('HyperparamTuner')


class HyperparamTuner:
    """
    Thực hiện Hyperparameter Tuning (Grid Search) hoặc huấn luyện mô hình cơ sở.
    """
    
    def __init__(self, model_name: str, 
                 scoring_metrics: Dict[str, str], 
                 refit_metric: str, 
                 param_grid: Dict[str, Any], 
                 gpu: int,
                 seed: int):
        
        self.model_name = model_name
        self.scoring_metrics = scoring_metrics
        self.refit_metric = refit_metric
        self.param_grid = param_grid
        self.seed = seed
        self.gpu = gpu
        # Sử dụng ModelFactory để tạo model cơ sở 
        self.model = ModelFactory.create_model(model_name, gpu, seed) 
        logger.info(f"Đã khởi tạo Base Model: {self.model_name}")

    def run_grid_search(self, X_train, y_train) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Thực hiện Grid Search hoặc fit model cơ sở.
        """
        if not self.param_grid:
            logger.info(f"{self.model_name} - Training BASE model (No Tuning).")
            self.model.fit(X_train, y_train)
            # Trả về các tham số mặc định của model cơ sở
            best_params = self.model.get_params()
            return self.model, best_params

        else: 
            logger.info(f"{self.model_name} - Running GridSearchCV...")
            grid = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                scoring=self.scoring_metrics,
                refit=self.refit_metric,
                cv=3,
                n_jobs=-1,
                verbose=2
            )
            # Đảm bảo y_train là Series/array
            y = y_train.squeeze() if hasattr(y_train, 'squeeze') else y_train
            
            grid.fit(X_train, y)
            
            best_model = grid.best_estimator_
            best_params = grid.best_params_

            return best_model, best_params