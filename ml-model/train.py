import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings

warnings.filterwarnings(action='ignore', category=RuntimeWarning)  # Ignore runtime warnings

def load_data(file_path):
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

def train_model(X_train, y_train, param_grid):
    # Create pipeline with feature selection, polynomial features, scaling, and logistic regression
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)),  # Feature selection
        ('polynomial_features', PolynomialFeatures(degree=2)),    # Polynomial features
        ('scaler', StandardScaler()),                             # Feature scaling
        ('logistic_regression', LogisticRegression(max_iter=1000)) # Logistic regression
    ])

    grid_search = GridSearchCV(pipe, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best cross-validation score: {grid_search.best_score_}')

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    return accuracy, cm, cr

if __name__ == "__main__":
    df = load_data(r'C:\Users\divya\OneDrive\Desktop\smart health diagnosis assistant\dataset\cleaned_pcos_data.csv')
    df = preprocess_data(df)  # Preprocess data by removing constant and low-variance features

    X = df.drop(columns=['PCOS (Y/N)'])
    y = df['PCOS (Y/N)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'feature_selection__k': [10, 20, 30, 'all'],  # Number of top features to select
        'logistic_regression__C': [0.01, 0.1, 1, 10, 100]
    }

    model = train_model(X_train, y_train, param_grid)
    accuracy, cm, cr = evaluate_model(model, X_test, y_test)

    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')
    print(f'Classification Report:\n{cr}')

    # Ensure the models directory exists
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join( 'logistic_regression_model.pkl'))
