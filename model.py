import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from data_loader import load_student_dataset

def build_and_train_pipeline():
    """
    Loads raw data, selects a strictly controlled set of user-interactive features,
    and returns a fitted machine learning training Pipeline object.
    """
    df = load_student_dataset()
    target = 'Post_Semester_GPA'
    
    # CRITICAL FIX: Explicitly define the features we collect from the UI.
    # This prevents any unexpected dataset columns from crashing the app.
    feature_cols = [
        'Major_Category', 'Year_of_Study', 'Pre_Semester_GPA', 
        'Weekly_GenAI_Hours', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 
        'Tool_Diversity', 'Paid_Subscription', 'Traditional_Study_Hours', 
        'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams'
    ]
    
    X = df[feature_cols]
    y = df[target]
    
    # Isolate feature columns by type
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessor configuration
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
    
    print("Training model pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline, X