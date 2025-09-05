# src/pipelines/train_pipeline.py
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTrasformation
from src.components.model_trainer import ModelTrainer

def main():
    train_data, test_data = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, preprocessor, encoder = DataTrasformation().initiate_data_transformation(
        train_data, test_data
    )
    summary = ModelTrainer().initiate_model_training(
        train_arr, test_arr, preprocessor, encoder,
        enable_shap=False  # flip on when needed
    )
    print(summary)

if __name__ == "__main__":
    main()
