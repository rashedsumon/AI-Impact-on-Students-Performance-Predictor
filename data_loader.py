import os
import glob
import pandas as pd
import kagglehub

def load_student_dataset():
    """
    Downloads the 'ai-impact-on-students' dataset from Kaggle via kagglehub,
    locates the raw CSV file inside the downloaded path, and returns it as a Pandas DataFrame.
    """
    # 1. Download the latest version from Kaggle using the requested snippet
    print("Downloading dataset from Kaggle...")
    download_path = kagglehub.dataset_download("laveshjadon/ai-impact-on-students")
    print("Path to dataset files:", download_path)
    
    # 2. Find any CSV file inside the returned path folder
    csv_files = glob.glob(os.path.join(download_path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found inside the downloaded directory: {download_path}")
        
    # Pick the first matched CSV file 
    target_csv = csv_files[0]
    print(f"Reading data file from: {target_csv}")
    
    # 3. Load into DataFrame
    df = pd.read_csv(target_csv)
    return df

if __name__ == "__main__":
    # Test script execution independently
    try:
        data = load_student_dataset()
        print("Dataset loaded successfully!")
        print(data.head(3))
    except Exception as e:
        print(f"An error occurred: {e}")