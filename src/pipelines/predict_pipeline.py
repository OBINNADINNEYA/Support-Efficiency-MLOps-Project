
import sys
import pandas as pd 
from src.exceptions import logging,CustomException
from src.utils import build_features_for_inference, load_object,X_COLS
import sys

class PredictPipeline:
    def __init__(self):
        self.model = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")
        self.encoder = load_object("artifacts/labelencoder.pkl")

    def predict(self, raw_input):
        try:
            df = raw_input if isinstance(raw_input, pd.DataFrame) else pd.DataFrame(raw_input)
            X = df[X_COLS] if all(c in df.columns for c in X_COLS) else build_features_for_inference(df)
            X_trans = self.preprocessor.transform(X)
            y_enc = self.model.predict(X_trans)
            return self.encoder.inverse_transform(y_enc.astype(int))
        except Exception as e:
            from src.exceptions import CustomException, sys
            raise CustomException(e, sys)



# Helps to map all the data we receive from the user to a form 
# we can pass as input for the model
import sys
import pandas as pd
from src.exceptions import CustomException

class CustomData:

    def __init__(self, 
        text_size: int,
        comment_count: int, 
        participants_count: int,
        first_response_minutes: float,
        first_response_missing: int,
    ):
        self.text_size = text_size
        self.comment_count = comment_count
        self.participants_count = participants_count
        self.first_response_minutes = first_response_minutes
        self.first_response_missing = first_response_missing

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "text_size": [self.text_size],
                "comment_count": [self.comment_count],
                "participants_count": [self.participants_count],
                "first_response_minutes": [self.first_response_minutes],
                "first_response_missing": [self.first_response_missing],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)




    


