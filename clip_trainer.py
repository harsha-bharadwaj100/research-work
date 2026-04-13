"""
GlassEngine CLIP Few-Shot Trainer
Trains a logistic regression on CLIP features from your curated dataset.
Run this ONCE to generate glassengine_clip_model.pkl
"""
import os
import glob
import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import torch
from transformers import CLIPProcessor, CLIPModel

# ---- Config ----
REAL_DIR = r"c:\D-sim\pyprojs\research-work\AI_img_testing\Real_images"
FAKE_DIR = r"c:\D-sim\pyprojs\research-work\AI_img_testing\DF_Images"
MODEL_OUT = r"c:\D-sim\pyprojs\research-work\backend\glassengine_clip_model.pkl"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def extract_clip_features(image_paths, model, processor, device):
    features = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                # Use vision_model directly to get a plain tensor (transformers v5 compatible)
                vision_outputs = model.vision_model(**inputs)
                feat = vision_outputs.pooler_output  # shape: [1, 768] — a clean tensor
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            features.append(feat.cpu().numpy().flatten())
            print(f"  ✓ {os.path.basename(path)}")
        except Exception as e:
            print(f"  ✗ {os.path.basename(path)}: {e}")
    return np.array(features)

def main():
    print("=== GlassEngine CLIP Few-Shot Trainer ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\nLoading CLIP model (downloading once if needed ~350MB)...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    real_paths = sorted(glob.glob(os.path.join(REAL_DIR, "*.*")))
    fake_paths = sorted(glob.glob(os.path.join(FAKE_DIR, "*.*")))

    print(f"\nExtracting CLIP features from {len(real_paths)} real images...")
    real_features = extract_clip_features(real_paths, model, processor, device)

    print(f"\nExtracting CLIP features from {len(fake_paths)} fake images...")
    fake_features = extract_clip_features(fake_paths, model, processor, device)

    X = np.vstack([real_features, fake_features])
    y = np.array([1] * len(real_features) + [0] * len(fake_features))  # 1=real, 0=fake

    print(f"\nDataset shape: X={X.shape}, y={y.shape}")

    # Leave-one-out style evaluation on this tiny dataset
    print("\n--- Leave-One-Out Cross Validation ---")
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    preds, trues = [], []
    for train_idx, test_idx in loo.split(X):
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
        ])
        clf.fit(X[train_idx], y[train_idx])
        preds.append(clf.predict(X[test_idx])[0])
        trues.append(y[test_idx][0])

    print(f"LOO Accuracy: {accuracy_score(trues, preds)*100:.1f}%")
    print(classification_report(trues, preds, target_names=["AI Deepfake", "Authentic"]))

    # Train final model on ALL data
    print("\n--- Training final model on ALL data ---")
    final_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    final_clf.fit(X, y)

    joblib.dump(final_clf, MODEL_OUT)
    print(f"\n✅ Model saved to {MODEL_OUT}")
    print("Now restart your FastAPI backend — it will auto-load the new model!")

if __name__ == "__main__":
    main()
