import os
import sys
import time
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import CustomException
from src.logger import logging
from src.components.data_transformation import *
from src.components.model_trainer import *

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = "artifacts"
    train_data_path: str = os.path.join("artifacts", "train.parquet")
    test_data_path: str  = os.path.join("artifacts", "test.parquet")

class DataIngestion:
    def __init__(self, src_parquet: str = "artifacts/train-00000-of-00127.parquet"):
        self.src_parquet = src_parquet
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component")
        try:

            t0 = time.time()
            df = pd.read_parquet(self.src_parquet, engine="pyarrow")
            logging.info(f"READ: {df.shape} in {time.time()-t0:.1f}s")

            t1 = time.time()
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"SPLIT: {time.time()-t1:.1f}s")
            logging.info(f"SPLIT: {train_set.shape} and {test_set.shape} in {time.time()-t0:.1f}s")

            t2 = time.time()
            train_set.to_parquet(self.ingestion_config.train_data_path, index=False,
                                engine="pyarrow", compression="zstd")
            logging.info(f"WRITE train: {time.time()-t2:.1f}s")

            t3 = time.time()
            test_set.to_parquet(self.ingestion_config.test_data_path, index=False,
                                engine="pyarrow", compression="zstd")
            logging.info(f"WRITE test : {time.time()-t3:.1f}s")

            logging.info(f"TOTAL: {time.time()-t0:.1f}s")

            logging.info("Ingestion of the data is complete")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_data, test_data = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, preprocessor,encoder = DataTrasformation().initiate_data_transformation(train_data, test_data)
    ModelTrainer().initiate_model_training(train_arr, test_arr, preprocessor,encoder)
