import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from data_loader import load_student_dataset

def build_and_train_pipeline():
    """
    Loads raw data and trains on a strictly controlled set of 11 features
    that match our Streamlit UI inputs exactly.
    """
    df = load_student_dataset()
    target = 'Post_Semester_GPA'
    
    # EXACT MATCH: Explicitly choosing the columns we take from the user interface
    feature_cols = [
        'Major_Category', 'Year_of_Study', 'Pre_Semester_GPA', 
        'Weekly_GenAI_Hours', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 
        'Tool_Diversity', 'Paid_Subscription', 'Traditional_Study_Hours', 
        'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams'
    ]
    
    # Filter dataset to match only these columns
    X = df[feature_cols]
    y = df[target]
    
    # Split feature tracking by type
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessor configurations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ])
    
    # Construct complete scikit-learn training pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    # Train the pipeline sequence
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline, X