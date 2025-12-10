import argparse
import os
import logging
import numpy as np
import pandas as pd
from src.training import TrainingPipeline, DataIOLoader
from src.data_processor import DataProcessor
from src.utils.data_config import save_processed_data, build_preprocess_config
# ============================================================================
# ARGUMENT PARSER

def parse_arguments():
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Pipeline huấn luyện mô hình hồi quy."
    )

    # ----- Data -----
    data_arg_group = argparser.add_argument_group("Data arguments")
    data_arg_group.add_argument('--X_train_path', type=str,
                                default='Data_processed/X_train.csv',
                                help='Path tới file X_train đã xử lý.')
    data_arg_group.add_argument('--X_test_path', type=str,
                                default='Data_processed/X_test.csv',
                                help='Path tới file X_test đã xử lý.')
    data_arg_group.add_argument('--y_train_path', type=str,
                                default='Data_processed/y_train.csv',
                                help='Path tới file y_train đã xử lý.')
    data_arg_group.add_argument('--y_test_path', type=str,
                                default='Data_processed/y_test.csv',
                                help='Path tới file y_test đã xử lý.')
    
    data_arg_group.add_argument('--data_input_path', type=str,
                                help='Path tới file CSV dữ liệu thô')
    data_arg_group.add_argument('--data_output_path', type=str,
                                default='data',
                                help='Path lưu file dữ liệu đã xử lý')
    data_arg_group.add_argument('--data_process', type=int,
                                default=0,
                                help='1 = bật pipeline tiền xử lý dữ liệu')
    data_arg_group.add_argument('--tuner', type=int,
                                default=0,
                                help='1 = bật hyperparameter tuning')

    # ----- Model -----
    model_arg_group = argparser.add_argument_group("Model arguments")
    model_arg_group.add_argument('--model_name', type=str,
                                 default='LinearRegression',
                                 choices=['LinearRegression', 'LGBMRegressor', 'XGBRegressor'],
                                 help='Tên model để train')
    model_arg_group.add_argument('--hyperparam_config_path', type=str,
                                 default='config/hyperparams.json',
                                 help='Path tới file JSON hyperparameters')

    # ----- Other -----
    data_arg_group.add_argument('--gpu', type=int,
                                default=0,  
                                choices=[0, 1], 
                                help='1 = Bật sử dụng GPU cho quá trình training, 0 = Tắt GPU (mặc định)')
    other_arg_group = argparser.add_argument_group("Other")
    other_arg_group.add_argument('--run_id', type=str,
                                 default="run_001",
                                 help='ID của lần chạy model')
    other_arg_group.add_argument('--savedir_models', type=str,
                                 default='outputs/models',
                                 help='Thư mục để lưu file .pkl model')
    other_arg_group.add_argument('--seed', type=int,
                                 default=42,
                                 help='Random seed')

    return argparser.parse_args()

# ============================================================================
# ORCHESTRATOR CLASS (NEW)

class ExperimentRunner:
    """
    Điều phối toàn bộ quy trình: Data Processing, Data Loading, Training Pipeline.
    """
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        
    def process_data(self):
        self.logger.info("\n========== [1] BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU ==========")

        input_path = self.args.data_input_path
        output_path = self.args.data_output_path
        
        if not input_path:
             self.logger.error("LỖI: Chưa cung cấp --data_input_path cho chế độ xử lý dữ liệu.")
             return

        self.logger.info(f"• Input path:  {input_path}")
        self.logger.info(f"• Output path: {output_path}")

        try:
            config = build_preprocess_config() 
            processor = DataProcessor(config=config) 

            df = processor.DataLoader(input_path)

            logger.info("→ Chạy pipeline xử lý dữ liệu...")
            result = processor.run_pipeline(
                df=df,
                target_col='price',
            )

            # Lưu dữ liệu đã xử lý
            os.makedirs(output_path, exist_ok=True)
            result["X_train"].to_csv(f"{output_path}/X_train.csv", index=False)
            result["X_test"].to_csv(f"{output_path}/X_test.csv", index=False)
            result["y_train"].to_csv(f"{output_path}/y_train.csv", index=False)
            result["y_test"].to_csv(f"{output_path}/y_test.csv", index=False)

            self.logger.info("→ Đã lưu dữ liệu đã xử lý.")
            self.logger.info("========== HOÀN TẤT TIỀN XỬ LÝ ==========\n")
            
        except Exception as e:
            self.logger.error("LỖI KHI TIỀN XỬ LÝ DỮ LIỆU.")
            self.logger.error(str(e))


    def execute_training_pipeline(self):
        self.logger.info("\n==========  NẠP DỮ LIỆU TRAIN/TEST ==========")

        #  Nạp dữ liệu (Sử dụng DataIOLoader)
        try:
            X_train, X_test, y_train, y_test = DataIOLoader.load_processed_data(self.args)
        except Exception:
            # Lỗi đã được log trong DataIOLoader
            return

        self.logger.info(f" Train size: {len(X_train):,}")
        self.logger.info(f" Test size : {len(X_test):,}")
        self.logger.info("========== HOÀN TẤT NẠP DỮ LIỆU ==========\n")

        # 2. Huấn luyện (Sử dụng TrainingPipeline)
        self.logger.info("========== [3] BẮT ĐẦU TRAINING PIPELINE ==========")
        
        # Chuyển args thành dictionary cho config
        run_config = vars(self.args)
        
        try:
            pipeline = TrainingPipeline(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                random_seed=self.args.seed,
                run_config=run_config
            )
            pipeline.run_pipeline()
            
        except Exception as e:
            self.logger.error("LỖI KHI CHẠY TRAINING PIPELINE")
            self.logger.error(str(e))
            return

        self.logger.info("========== HOÀN TẤT TRAINING PIPELINE ==========\n")


# ============================================================================
# LOGGER SETUP 

def setup_logger(log_file_path="outputs/logs/pipeline.log"):
    """
    Lưu logging ra file + hiển thị terminal.

    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# ============================================================================
# MAIN EXECUTION

if __name__ == "__main__":
    args = parse_arguments()
    np.random.seed(args.seed)

    # Cấu hình log file dựa trên chế độ chạy
    if args.data_process:
        log_file = f"outputs/logs/data_processing.log"
        logger = setup_logger(log_file)
        runner = ExperimentRunner(args, logger)
        runner.process_data()
    else:
        log_file = f"outputs/logs/{args.run_id}_{args.model_name}.log"
        logger = setup_logger(log_file)

        logger.info(f"BẮT ĐẦU PIPELINE | RUN ID: {args.run_id}")

        logger.info("\n========== THÔNG SỐ CHẠY PIPELINE ==========")
        for k, v in vars(args).items():
            logger.info(f"{k}: {v}")
        logger.info("============================================\n")

        runner = ExperimentRunner(args, logger)
        runner.execute_training_pipeline()