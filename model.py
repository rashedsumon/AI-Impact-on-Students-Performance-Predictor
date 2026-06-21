import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from data_loader import load_student_dataset

def build_and_train_pipeline():
    """
    Loads raw data, splits features, applies basic preprocessing,
    and returns a fitted machine learning training Pipeline object.
    """
    # Load dataset through data_loader script
    df = load_student_dataset()
    
    # Define features (X) and target label (y) based on dataset columns
    # We will predict Post_Semester_GPA
    target = 'Post_Semester_GPA'
    
    # Ensure standard expected column drops if matching Kaggle dataset syntax
    ignore_cols = [target]
    if 'Student_ID' in df.columns:
        ignore_cols.append('Student_ID')
        
    X = df.drop(columns=ignore_cols)
    y = df[target]
    
    # Isolate feature columns by type
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Use Ordinal Encoder for user categories (e.g., Major, Skill level, Use case)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ])
    
    # Package preprocessing and regressor algorithm inside an execution Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    # Split into train/test sets to evaluate execution score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model pipeline... (this might take a few moments)")
    model_pipeline.fit(X_train, y_train)
    
    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    print(f"Model Training Complete! Train R²: {train_score:.2f} | Test R²: {test_score:.2f}")
    
    return model_pipeline, X

if __name__ == "__main__":
    # Test file execution standalone
    build_and_train_pipeline()