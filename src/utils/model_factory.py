# src/utils/model_factory.py
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor 
from sklearn.base import BaseEstimator
from typing import Type

class ModelFactory:
    """
    Factory Class chịu trách nhiệm khởi tạo các Estimator (mô hình).
    """

    @staticmethod
    def create_model(model_name: str, gpu: int, seed: int) -> BaseEstimator:
        """
        Khởi tạo và trả về instance của mô hình dựa trên tên.
        """
        if model_name == 'LinearRegression':
            return LinearRegression()
        
        elif model_name == 'LGBMRegressor':
            if gpu == 1:
                return LGBMRegressor(
                                    random_state=seed,
                                    device='gpu',
                                    n_jobs=-1,
                                )
            else:
                return LGBMRegressor(
                    random_state=seed,
                    n_jobs=-1,

                )
            
        elif model_name == 'XGBRegressor':
                return XGBRegressor(
                    objective='reg:squarederror', 
                    random_state=seed, 
                    n_jobs=-1
                )
            

        # Có thể thêm các mô hình khác (ví dụ: SVR, RandomForest,...)
        # elif model_name == 'SVR':
        #     return SVR()

        else:
            raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ trong ModelFactory.")