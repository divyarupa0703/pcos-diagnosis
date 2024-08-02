import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

def load_model(model_path):
    return joblib.load(model_path)

def load_train_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Remove constant features
    df = df.loc[:, df.apply(pd.Series.nunique) != 1]

    # Remove low-variance features
    variance_threshold = 0.01  # Adjust as needed
    variances = df.var()
    df = df.loc[:, variances > variance_threshold]

    # Drop rows with NaN or infinite values
    df = df.replace([float('inf'), -float('inf')], pd.NA)
    df = df.dropna()

    return df

def preprocess_user_input(user_input, feature_columns):
    user_input_df = pd.DataFrame([user_input])
    
    # Ensure all required columns are present
    missing_cols = set(feature_columns) - set(user_input_df.columns)
    for col in missing_cols:
        user_input_df[col] = 0  # or np.nan if you prefer

    # Reorder columns to match training data
    user_input_df = user_input_df[feature_columns]
    
    return user_input_df

def predict(model, user_input_df):
    # Extract steps from the pipeline
    feature_selection = model.named_steps['feature_selection']
    poly_features = model.named_steps['polynomial_features']
    
    # Feature selection
    X_selected = feature_selection.transform(user_input_df)
    
    # Polynomial features
    X_poly = poly_features.transform(X_selected)
    
    # Prediction
    prediction = model.named_steps['logistic_regression'].predict(X_poly)
    return prediction

if __name__ == "__main__":
    model_path = 'logistic_regression_model.pkl'
    model = load_model(model_path)

    # Load training data to get the feature columns
    train_data_path = r'C:\Users\divya\OneDrive\Desktop\smart health diagnosis assistant\dataset\cleaned_pcos_data.csv'
    df_train = load_train_data(train_data_path)
    
    # Preprocess training data to get feature columns
    df_train = preprocess_data(df_train)  # Ensure this function is available
    feature_columns = df_train.drop(columns=['PCOS (Y/N)']).columns

    # Example: User input (replace with actual user input collection)
    user_input = {
        'Age (yrs)': 28,
        'Weight (Kg)': 44.6,
        'Height(Cm)': 152,
        'BMI': 19.3,
        'Blood Group': 15,
        'Pulse rate(bpm)': 78,
        'RR (breaths/min)': 22,
        'Hb(g/dl)': 10.48,
        'Cycle(R/I)': 2,
        'Cycle length(days)': 5,
        'Marraige Status (Yrs)': 7,
        'Pregnant(Y/N)': 0,
        'No. of aborptions': 0,
        ' I beta-HCG(mIU/mL)': 1.99,
        'II beta-HCG(mIU/mL)': 1.99,
        'FSH(mIU/mL)':7.95,
        'LH(mIU/mL)': 3.68,
        'FSH/LH': 2.160326087,
        'Hip(inch)': 36,
        'Waist(inch)': 30,
        'Waist:Hip Ratio': 0.83333,
        'TSH (mIU/L)': 0.68,
        'AMH(ng/mL)': 2.07,
        'PRL(ng/mL)': 45.16,
        'Vit D3 (ng/mL)': 17.1,
        'PRG(ng/mL)': 0.57,
        'RBS(mg/dl)': 92,
        'Weight gain(Y/N)': 0,
        'hair growth(Y/N)': 0,
        'Skin darkening (Y/N)': 0,
        'Hair loss(Y/N)': 0,
        'Pimples(Y/N)': 0,
        'Fast food (Y/N)': 1,
        'Reg.Exercise(Y/N)': 0,
        'BP _Systolic (mmHg)': 110,
        'BP _Diastolic (mmHg)': 80,
        'Follicle No. (L)': 3,
        'Follicle No. (R)': 3,
        'Avg. F size (L) (mm)': 18,
        'Avg. F size (R) (mm)': 18,
        'Endometrium (mm)': 8.5
    }

    user_input_df = preprocess_user_input(user_input, feature_columns)
    prediction = predict(model, user_input_df)
    print(f'Predicted PCOS status: {"Yes" if prediction[0] == "Y" else "No"}')
