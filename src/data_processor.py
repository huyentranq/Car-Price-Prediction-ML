import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer,
                                   OneHotEncoder, OrdinalEncoder, TargetEncoder)
from sklearn.model_selection import train_test_split
from src.utils.data_config import PreprocessConfig
 
# from data_config import *
class RowCleaner(BaseEstimator, TransformerMixin):
    """
    Stateless transformers before train-test split.
    """
    def __init__(self, config: PreprocessConfig):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df = self._remove_duplicates(df)
        df = self._extract_numbers(df)
        df = self._convert_datetime(df)
        df = self._drop_columns(df)
        df = self._drop_na_rows(df)
        df = self._apply_hard_filters(df)

        return df


    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.deduplicate:
            original_len = len(df)
            df = df.drop_duplicates()
            dropped_count = original_len - len(df)
            if dropped_count > 0:
                print(f"[RowCleaner] Dropped {dropped_count} duplicate rows")
        return df

    def _extract_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.config.extract_num_cols:
            if col in df.columns:
                extracted = df[col].astype(str).str.extract(r'(\d+)')[0]
                df[col] = pd.to_numeric(extracted, errors='coerce')
        return df

    def _convert_datetime(self, df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
        for col in self.config.datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[f'{col}_year'] = df[col].dt.year
            if drop_original:
                df.drop(col, axis = 1, inplace = True)
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.drop_cols:
            return df.drop(columns=self.config.drop_cols, errors='ignore')
        return df

    def _drop_na_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.drop_na_cols:
            cols = [c for c in self.config.drop_na_cols if c in df.columns]
            if cols:
                original_len = len(df)
                df = df.dropna(subset=cols)
                dropped_count = original_len - len(df)
                if dropped_count > 0:
                    print(f"[RowCleaner] Dropped {dropped_count} rows (NaN in {cols})")
        return df

    def _apply_hard_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (min_v, max_v) in self.config.hard_filters.items():
            if col in df.columns:
                if min_v is not None: df = df[df[col] >= min_v]
                if max_v is not None: df = df[df[col] <= max_v]
        return df
    

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateful transformers after train-test split.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self._transformer_chain: Optional[Pipeline] = None
        self._feature_names_out: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self._transformer_chain = self._build_transformer_chain()
        self._transformer_chain.fit(X, y)

        if hasattr(self._transformer_chain, 'get_feature_names_out'):
            try:
                self._feature_names_out = list(self._transformer_chain.get_feature_names_out())
            except Exception:
                self._feature_names_out = list(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._transformer_chain is None:
            raise RuntimeError("FeatureEngineer chưa được fit. Hãy gọi fit() trước.")
        return self._transformer_chain.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out

    def _build_transformer_chain(self) -> Pipeline:
        steps = []

        # 1. Pre-impute Custom Functions
        if self.config.custom_functions_pre_impute:
            steps.append(('pre_custom', self._make_func_transformer(self.config.custom_functions_pre_impute)))

        # 2. Imputation
        imputer = self._build_imputer_step()
        if imputer: steps.append(('imputation', imputer))

        # 3. Post-impute Custom Functions
        if self.config.custom_functions_post_impute:
            steps.append(('post_custom', self._make_func_transformer(self.config.custom_functions_post_impute)))

        # 4. Feature Processing (Encode/Scale)
        processor = self._build_processing_step()
        if processor: steps.append(('feature_processing', processor))

        # Trả về Pipeline (đóng vai trò là transformer chain)
        pipeline = Pipeline(steps)
        pipeline.set_output(transform="pandas")
        return pipeline

    def _make_func_transformer(self, funcs: List[Callable]):
        def process_funcs(df):
            for func in funcs:
                df = func(df)
            return df
        return FunctionTransformer(process_funcs, validate=False)

    def _build_imputer_step(self):
        transformers = []
        strats = self.config.impute_strategies

        if strats['median']:
            transformers.append(('median', SimpleImputer(strategy='median'), strats['median']))
        if strats['mean']:
            transformers.append(('mean', SimpleImputer(strategy='mean'), strats['mean']))
        if strats['mode']:
            transformers.append(('mode', SimpleImputer(strategy='most_frequent'), strats['mode']))
        for col, val in strats['constant'].items():
            transformers.append((f'const_{col}', SimpleImputer(strategy='constant', fill_value=val), [col]))

        return ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False) if transformers else None

    def _build_processing_step(self):
        transformers = []

        enc_map = {
            'target': TargetEncoder(target_type='continuous', smooth='auto', random_state=42),
            'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)
        }

        for col, strategy in self.config.encode_cols.items():
            if strategy in enc_map:
                transformers.append((f'enc_{strategy}_{col}', enc_map[strategy], [col]))

        # Scaling
        scale_map = {
            'log': FunctionTransformer(np.log1p, validate=False),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        for col, strategy in self.config.scale_cols.items():
            if strategy in scale_map:
                transformers.append((f'scale_{strategy}_{col}', scale_map[strategy], [col]))

        return ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False) if transformers else None
    


class DataProcessor:
    """
    Quản lý toàn bộ quy trình tiền xử lý dữ liệu
    """
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.row_cleaner = RowCleaner(config)
        self.feature_engineer = FeatureEngineer(config)
        self._is_fitted = False

    def DataLoader(self, path: str) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file, hỗ trợ CSV/JSON/Excel.

        """
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.json'):
            return pd.read_json(path)
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    def run_pipeline(self, df: pd.DataFrame,
                     target_col: str,
                     test_size=0.2,
                     random_state=42,
                     target_transform_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Chạy quy trình huấn luyện.
        """
        print(f"--- Bắt đầu quy trình huấn luyện (Shape: {df.shape}) ---")

        # Step 1: CLEANING
        df_clean = self.row_cleaner.transform(df)
        print(f"1. Row Cleaning done. Shape: {df_clean.shape}")

        # Step 2: SPLIT
        if target_col not in df_clean.columns:
            raise ValueError(f"Missing target column: {target_col}")

        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"2. Split done. Train: {len(X_train)}, Test: {len(X_test)}")

        if target_transform_func:
            y_train = target_transform_func(y_train)
            y_test = target_transform_func(y_test)
            print(f"   -> Target transformed using {target_transform_func.__name__}")

        # Step 3: FEATURE ENGINEERING
        self.feature_engineer.fit(X_train, y_train)
        self._is_fitted = True

        # Transform Train và Test
        X_train_proc = self.feature_engineer.transform(X_train)
        X_test_proc = self.feature_engineer.transform(X_test)
        print("3. Feature Engineering done.")

        return {
            'X_train': X_train_proc,
            'X_test': X_test_proc,
            'y_train': y_train,
            'y_test': y_test,
            'engineer': self.feature_engineer
        }

    def predict_new_data(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Dự đoán dữ liệu mới.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline chưa được fit. Hãy chạy run_pipeline trước.")

        df_clean = self.row_cleaner.transform(df_new)

        if df_clean.empty:
            return pd.DataFrame()

        return self.feature_engineer.transform(df_clean)


