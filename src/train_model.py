import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train():
    raw_data_dir = '../data/raw'
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(raw_data_dir, '*.csv'))
    if not csv_files:
        print("Error: No CSV files found.")
        return
        
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        label = os.path.basename(file).split('.')[0]
        df['label'] = label
        all_data.append(df)
        
    final_df = pd.concat(all_data, ignore_index=True)
    X = final_df.drop('label', axis=1)
    y = final_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    model_path = os.path.join(model_dir, 'sign_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()