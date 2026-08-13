import sys
from src.exception_handler import CustomException
from src.logger import logging
from src.utils import load_object
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64



class PredictionPipeline:
    def __init__(self,Age,Gender:str,Tenure,CreditScore,
                 IsActiveMember,Geography,NumOfProducts) :
        # all the custom values
        self.Age=Age
        self.Gender=Gender
        self.Tenure=Tenure
        self.CreditScore=CreditScore
        self.IsActiveMember=IsActiveMember
        self.Geography=Geography
        self.NumOfProducts=NumOfProducts

    def _get_dataframe(self):
        """
        Creates a DataFrame from the input data.
        """
        try:
            data_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Tenure": [self.Tenure],
                "CreditScore": [self.CreditScore],
                "IsActiveMember": [self.IsActiveMember],
                "Geography": [self.Geography],
                "NumOfProducts": [self.NumOfProducts],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e,sys)
    
    def predict(self):
        logging.info("Predict method invoked.")
        """
        Makes predictions using the trained model.
        """
        try:
            logging.info("Loading the model.")
            model=load_object("artifacts\model.pkl")

            logging.info("Loading the Preprocessor.")
            preprocessor=load_object("artifacts\preprocessor.pkl")

            data=self._get_dataframe()
            prediction_data=preprocessor.transform(data)
            logging.info("Processed the data.")
            result=model.predict(prediction_data)
            return result

        except Exception as e:
            raise CustomException(e,sys)

    def predict_and_explain(self):
        logging.info("Predict and explain method invoked.")
        try:
            logging.info("Loading the model and preprocessor for explanation.")
            model=load_object("artifacts\model.pkl")
            preprocessor=load_object("artifacts\preprocessor.pkl")

            data=self._get_dataframe()
            prediction_data=preprocessor.transform(data)
            logging.info("Processed the data for explanation.")
            
            result=model.predict(prediction_data)
            
            # Explain with SHAP
            logging.info("Generating SHAP explanation.")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(prediction_data)
            
            # Handle both classification array types
            if isinstance(shap_values, list):
                shap_val = shap_values[1]
            else:
                shap_val = shap_values

            # Try to get feature names, fallback to generic if not possible
            try:
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                feature_names = [f"Feature {i}" for i in range(prediction_data.shape[1])]
                
            plt.figure(figsize=(8, 4))
            shap.summary_plot(shap_val, prediction_data, feature_names=feature_names, plot_type="bar", show=False)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return result, img_base64

        except Exception as e:
            raise CustomException(e,sys)

