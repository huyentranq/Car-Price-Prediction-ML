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



class PreprocessConfig:
    """Configuration class for data preprocessing"""
    def __init__(self):
        self._deduplicate: bool = True
        self._drop_cols: List[str] = []
        self._drop_na_cols: List[str] = []
        self._datetime_cols: List[str] = []
        self._extract_num_cols: List[str] = []
        self._hard_filters: Dict[str, tuple] = {}

        self._impute_strategies = {
            'median': [], 'mean': [], 'mode': [], 'constant': {}
        }
        self._encode_cols: Dict[str, str] = {}
        self._scale_cols: Dict[str, str] = {}

        self._custom_functions_pre_impute: List[Callable] = []
        self._custom_functions_post_impute: List[Callable] = []


    @property
    def deduplicate(self) -> bool: return self._deduplicate
    @property
    def drop_cols(self) -> List[str]: return self._drop_cols
    @property
    def drop_na_cols(self) -> List[str]: return self._drop_na_cols
    @property
    def datetime_cols(self) -> List[str]: return self._datetime_cols
    @property
    def extract_num_cols(self) -> List[str]: return self._extract_num_cols
    @property
    def hard_filters(self) -> Dict[str, tuple]: return self._hard_filters
    @property
    def impute_strategies(self) -> Dict: return self._impute_strategies
    @property
    def encode_cols(self) -> Dict: return self._encode_cols
    @property
    def scale_cols(self) -> Dict: return self._scale_cols
    @property
    def custom_functions_pre_impute(self) -> List[Callable]: return self._custom_functions_pre_impute
    @property
    def custom_functions_post_impute(self) -> List[Callable]: return self._custom_functions_post_impute

    # Column Operations
    def set_deduplicate(self, deduplicate: bool) -> 'PreprocessConfig':
        self._deduplicate = deduplicate
        return self

    def add_drop_cols(self, *cols: str) -> 'PreprocessConfig':
        self.drop_cols.extend(cols)
        return self

    def add_dropna_strategy(self, *cols: str) -> 'PreprocessConfig':
        self.drop_na_cols.extend(cols)
        return self

    def add_datetime_convert(self, *cols: str) -> 'PreprocessConfig':
        self.datetime_cols.extend(cols)
        return self

    def add_extract_number(self, *cols: str) -> 'PreprocessConfig':
        self._extract_num_cols.extend(cols)
        return self

    # Filtering
    def add_hard_filter(self, col: str, min_val: Optional[float] = None,
                       max_val: Optional[float] = None) -> 'PreprocessConfig':
        if min_val is not None and max_val is not None:
            if min_val > max_val:
                raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")

        self.hard_filters[col] = (min_val, max_val)
        return self

    # Imputation
    def add_fillna_strategy(self, col: str, strategy: str = 'median',
                           value: Any = None) -> 'PreprocessConfig':
        valid_strategies = ['median', 'mean', 'mode', 'constant']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")

        if strategy == 'constant':
            if value is None:
                raise ValueError("Must provide 'value' when strategy='constant'")
            self.impute_strategies['constant'][col] = value

        elif strategy in ['median', 'mean', 'mode']:
            self.impute_strategies[strategy].append(col)

        return self

    # Encoding and Scaling
    def add_encode_strategy(self, col: str, strategy: str) -> 'PreprocessConfig':
        valid_strategies = ['target', 'onehot', 'ordinal']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
        self.encode_cols[col] = strategy
        return self

    def add_scale_strategy(self, col: str, strategy: str) -> 'PreprocessConfig':
        valid_strategies = ['log', 'standard', 'minmax', 'robust']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
        self.scale_cols[col] = strategy
        return self

    def add_custom_function(self, func: Callable,
                           stage: str = 'pre_impute') -> 'PreprocessConfig':
        if not callable(func):
            raise TypeError(f"Expected callable, got {type(func)}")
        if stage == 'pre_impute':
            self.custom_functions_pre_impute.append(func)
        elif stage == 'post_impute':
            self.custom_functions_post_impute.append(func)
        else:
            raise ValueError(f"Invalid stage '{stage}'. Must be 'pre_impute' or 'post_impute'")

        return self

    # === Utility Methods ===

    def get_all_configured_columns(self) -> set:
        """Return a set of all configured columns"""
        cols = set()
        cols.update(self.drop_cols)
        cols.update(self.drop_na_cols)
        cols.update(self.datetime_cols)
        cols.update(self.hard_filters.keys())

        for strategy, value in self.impute_strategies.items():
            if isinstance(value, list):
                cols.update(value)
            elif isinstance(value, dict):
                cols.update(value.keys())

        cols.update(self.encode_cols.keys())
        cols.update(self.scale_cols.keys())
        return cols

    def to_dict(self) -> dict:
        """Export config to dictionary for serialization"""
        return {
            'drop_cols': self.drop_cols,
            'drop_na_cols': self.drop_na_cols,
            'datetime_cols': self.datetime_cols,
            'hard_filters': self.hard_filters,
            'impute_strategies': self.impute_strategies,
            'encode_cols': self.encode_cols,
            'scale_cols': self.scale_cols,
            # Custom functions cannot be serialized easily
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PreprocessConfig':
        """Load config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key) and key not in ['custom_functions_pre_impute',
                                                      'custom_functions_post_impute']:
                setattr(config, key, value)
        return config

    def __repr__(self) -> str:
        """Readable representation"""
        lines = [
            f"PreprocessConfig(",
            f"  drop_cols={len(self.drop_cols)} columns",
            f"  drop_na_cols={len(self.drop_na_cols)} columns",
            f"  datetime_cols={len(self.datetime_cols)} columns",
            f"  hard_filters={len(self.hard_filters)} filters",
            f"  impute_strategies={len(self.impute_strategies)} strategies",
            f"  encode_cols={len(self.encode_cols)} columns",
            f"  scale_cols={len(self.scale_cols)} columns",
            f"  custom_functions={len(self.custom_functions_pre_impute) + len(self.custom_functions_post_impute)} functions",
            ")"
        ]
        return "\n".join(lines)
    

def save_processed_data(X_train, X_test, y_train, y_test, args):
    # os.makedirs("data", exist_ok=True)

    X_train.to_csv(args.X_train_path, index=False)
    X_test.to_csv(args.X_test_path, index=False)
    y_train.to_csv(args.y_train_path, index=False)
    y_test.to_csv(args.y_test_path, index=False)

    # logger.info("💾 Saved processed train/test files.")

def calculate_age_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính tuổi xe từ cột 'year'.
    """
    if 'year' not in df.columns:
        return df

    df = df.copy()
    df['age'] = df['posting_date_year'] - df['year']
    df.loc[df['age'] < 0, 'age'] = 0
    df.drop(columns = ['year','posting_date_year'], axis = 1, inplace = True)

    return df

def build_preprocess_config():
    conf = (
    PreprocessConfig()
    # A. Config cho RowCleaner (Sơ chế)
    .add_drop_cols('id', 'url', 'region_url', 'image_url', 'county', 'size', 'VIN', 'description', 'region')
    .add_extract_number('cylinders')
    .add_dropna_strategy('posting_date', 'year','model') # Xóa dòng nếu thiếu year, model
    .add_datetime_convert('posting_date') # Sẽ tạo ra cột year để lọc

    .add_hard_filter('price', 100, 100000)
    .add_hard_filter('year', 1981, 2021)
    .add_hard_filter('odometer', 0, 500000)

    # B. Config cho Pipeline (Feature Engineering)
    .add_custom_function(calculate_age_logic, stage='pre_impute')


    # Imputation
    .add_fillna_strategy('fuel','mode')
    .add_fillna_strategy('odometer','median')
    .add_fillna_strategy('transmission','mode')
    .add_fillna_strategy('title_status','mode')
    .add_fillna_strategy('cylinders','mode')
    .add_fillna_strategy('drive','constant','unknown')
    .add_fillna_strategy('type','constant','unknown')
    .add_fillna_strategy('manufacturer', 'constant', 'unknown')
    .add_fillna_strategy('paint_color','constant','unknown')
    .add_fillna_strategy('lat','median')
    .add_fillna_strategy('long','median')

    # Encoding
    .add_encode_strategy('condition', 'ordinal')
    .add_encode_strategy('manufacturer', 'target')
    .add_encode_strategy('model', 'target')
    .add_encode_strategy('state', 'target')
    .add_encode_strategy('fuel', 'onehot')
    .add_encode_strategy('title_status', 'onehot')
    .add_encode_strategy('transmission', 'onehot')
    .add_encode_strategy('drive', 'onehot')
    .add_encode_strategy('type', 'target')
    .add_encode_strategy('paint_color', 'target')

    # Scaling
    #
    .add_scale_strategy('odometer', 'standard')
    .add_scale_strategy('age', 'standard')
    .add_scale_strategy('lat', 'standard')
    .add_scale_strategy('long', 'standard')
    .add_scale_strategy('cylinders','standard')
    # '''
    )

    return conf
