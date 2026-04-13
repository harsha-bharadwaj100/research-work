import os
import cv2
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ---------------------------------------------------------
# 1. Feature Extraction Algorithms (The 4 Pillars)
# ---------------------------------------------------------

def extract_phase(channel):
    f_transform = np.fft.fft2(channel)
    f_shift = np.fft.fftshift(f_transform)
    return np.angle(f_shift)

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (256, 256)) # Faster processing for large datasets
    
    # Pillar 1 & 2 Setup
    b, g, r = cv2.split(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. PMR (Phase Peak-to-Median Ratio)
    phase_b, phase_g, phase_r = extract_phase(b), extract_phase(g), extract_phase(r)
    diff_rg = np.arctan2(np.sin(phase_r - phase_g), np.cos(phase_r - phase_g))
    diff_rb = np.arctan2(np.sin(phase_r - phase_b), np.cos(phase_r - phase_b))
    diff_gb = np.arctan2(np.sin(phase_g - phase_b), np.cos(phase_g - phase_b))
    
    bins = 100
    hist_rg, _ = np.histogram(diff_rg.ravel(), bins=bins, range=(-np.pi, np.pi))
    hist_rb, _ = np.histogram(diff_rb.ravel(), bins=bins, range=(-np.pi, np.pi))
    hist_gb, _ = np.histogram(diff_gb.ravel(), bins=bins, range=(-np.pi, np.pi))
    hist_combined = hist_rg + hist_rb + hist_gb
    
    center_idx = bins // 2
    peak_value = np.sum(hist_combined[center_idx-3:center_idx+4])
    background_mask = np.ones(bins, dtype=bool)
    background_mask[center_idx-3:center_idx+4] = False
    median_background = np.median(hist_combined[background_mask])
    pmr = peak_value / (median_background + 1e-5)
    
    # 2. Global Entropy
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.nan_to_num(np.log2(hist + 1e-7))
    global_entropy = -np.sum(hist * logs)
    
    # 3. High-Frequency Noise Residual (PRNU proxy variance)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    noise_variance = np.var(laplacian)
    
    # 4. Error Level Analysis (ELA) Mean
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    ela_diff = cv2.absdiff(img, decimg)
    ela_score = np.mean(ela_diff)
    
    return {
        'pmr': float(pmr),
        'entropy': float(global_entropy),
        'noise_variance': float(noise_variance),
        'ela_score': float(ela_score)
    }

# ---------------------------------------------------------
# 2. Dataset Processing Pipeline
# ---------------------------------------------------------

def process_dataset_to_csv(real_dir, fake_dir, csv_output="glassengine_features.csv"):
    data = []
    
    print(f"Extracting features from {real_dir}...")
    real_images = glob.glob(os.path.join(real_dir, '*.*'))
    for img_path in real_images:
        features = extract_features(img_path)
        if features:
            features['label'] = 1 # 1 for Authentic / Real
            data.append(features)
            
    print(f"Extracting features from {fake_dir}...")
    fake_images = glob.glob(os.path.join(fake_dir, '*.*'))
    for img_path in fake_images:
        features = extract_features(img_path)
        if features:
            features['label'] = 0 # 0 for Fake / AI Deepfake
            data.append(features)
            
    df = pd.DataFrame(data)
    df.to_csv(csv_output, index=False)
    print(f"Successfully saved {len(data)} image features to {csv_output}!")
    return df

# ---------------------------------------------------------
# 3. Model Training (Run this in Colab / Kaggle)
# ---------------------------------------------------------

def train_model(csv_path="glassengine_features.csv"):
    print("\n--- Training GlassEngine AI Model ---")
    df = pd.read_csv(csv_path)
    
    X = df[['pmr', 'entropy', 'noise_variance', 'ela_score']]
    y = df['label']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    # RF excels at finding non-linear relationships between mathematical pillars
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    print("\nModel Evaluation Accuracy:")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['AI Deepfake', 'Authentic']))
    
    # Understand which pillar was most important
    importances = model.feature_importances_
    features = X.columns
    print("\nFeature Importances (Which Pillar Mattered Most?):")
    for feat, imp in zip(features, importances):
        print(f" - {feat}: {imp*100:.1f}%")
        
    # Save Model
    joblib.dump(model, 'glassengine_rf_model.pkl')
    print("\nModel saved as 'glassengine_rf_model.pkl'. You can load this in your local FastAPI backend!")

if __name__ == "__main__":
    # INSTRUCTIONS FOR KAGGLE/COLAB:
    # 1. Upload your dataset into two folders: 'dataset/real/' and 'dataset/fake/'
    # 2. Update the paths below.
    # 3. Run this entire script.
    
    REAL_IMAGES_DIR = "./dataset/real"
    FAKE_IMAGES_DIR = "./dataset/fake"
    
    # If paths exist, build the CSV and train. 
    if os.path.exists(REAL_IMAGES_DIR) and os.path.exists(FAKE_IMAGES_DIR):
        process_dataset_to_csv(REAL_IMAGES_DIR, FAKE_IMAGES_DIR)
        train_model()
    else:
        print("Please set up the dataset directories: './dataset/real' and './dataset/fake' to run this script.")
