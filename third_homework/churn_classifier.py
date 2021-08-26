import joblib
import pandas as pd


class ChurnClassifier(object):
    def __init__(self):
        self.model = joblib.load('random_forest_clf1.pkl')

    def predict_customer_churn(self, predictable_df):
        try:
            predictable_df['Churn'] = self.model.predict(predictable_df)
            predicted_df = predictable_df
            return predicted_df
        except:
            print('Failed to predict')
            # return None
