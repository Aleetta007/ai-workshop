import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    """
    Load the PCOS dataset and perform preprocessing.
    
    Args:
        filepath (str): Path to the CSV file containing the dataset.
    
    Returns:
        tuple: Preprocessed features (X), target (y), and feature names.
    """
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Assuming the dataset has columns: Age, BMI, Cycle_Length, Acne, Hair_Growth, Insulin, LH_FSH_Ratio, Weight_Gain, Irregular_Periods, PCOS
    # PCOS is the target variable (1 for at risk, 0 for not)
    
    # Handle missing values - fill with median for numerical, mode for categorical
    numerical_cols = ['Age', 'BMI', 'Cycle_Length', 'Insulin', 'LH_FSH_Ratio']
    categorical_cols = ['Acne', 'Hair_Growth', 'Weight_Gain', 'Irregular_Periods']
    
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = df.drop('PCOS', axis=1)
    y = df['PCOS']
    
    # Normalize numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, X.columns.tolist()

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random state for reproducibility
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        random_state: Random state for reproducibility
    
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using various metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics

def save_model(model, filepath):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a saved model from disk.
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Loaded model
    """
    return joblib.load(filepath)

def plot_feature_importance(model, feature_names, filepath=None):
    """
    Plot feature importance of the model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        filepath: Optional path to save the plot
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()