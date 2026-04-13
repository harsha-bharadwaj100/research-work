import os
import sys
import cv2
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib

# --- CLIP Few-Shot Classifier (Primary Engine) ---
clip_model = None
clip_processor = None
clip_classifier = None  # the trained sklearn logistic regression pipeline

CLIP_CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "glassengine_clip_model.pkl")

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image as PILImage

    print("[GlassEngine] Loading CLIP encoder (openai/clip-vit-base-patch32)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print("[GlassEngine] CLIP encoder ready.")
except Exception as e:
    print(f"[GlassEngine] CLIP load error: {e}")

if os.path.exists(CLIP_CLASSIFIER_PATH):
    try:
        clip_classifier = joblib.load(CLIP_CLASSIFIER_PATH)
        print(f"[GlassEngine] CLIP classifier loaded from {CLIP_CLASSIFIER_PATH}")
    except Exception as e:
        print(f"[GlassEngine] CLIP classifier load error: {e}")
else:
    print(f"[GlassEngine] No CLIP classifier found at {CLIP_CLASSIFIER_PATH}. Run clip_trainer.py first.")

def get_clip_features(pil_image):
    """Extract normalized CLIP features from a PIL image."""
    if clip_model is None or clip_processor is None:
        return None
    inputs = clip_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(**inputs)
        feat = vision_outputs.pooler_output
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().flatten()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_phase(channel):
    f_transform = np.fft.fft2(channel)
    f_shift = np.fft.fftshift(f_transform)
    return np.angle(f_shift)

def azimuthal_average(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)**2
    y, x = np.indices(mag.shape)
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(int)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    radial_prof = tbin / np.maximum(nr, 1)
    return radial_prof

def calculate_icpc(image):
    b, g, r = cv2.split(image)
    phase_b = extract_phase(b)
    phase_g = extract_phase(g)
    phase_r = extract_phase(r)
    
    diff_rg = phase_r - phase_g
    diff_rb = phase_r - phase_b
    diff_gb = phase_g - phase_b
    
    diff_rg = np.arctan2(np.sin(diff_rg), np.cos(diff_rg))
    diff_rb = np.arctan2(np.sin(diff_rb), np.cos(diff_rb))
    diff_gb = np.arctan2(np.sin(diff_gb), np.cos(diff_gb))
    
    return diff_rg, diff_rb, diff_gb

def fast_local_entropy(image_gray, window_size=5):
    try:
        from skimage.filters.rank import entropy
        from skimage.morphology import disk
        return entropy(image_gray, disk(window_size))
    except ImportError:
        img_float = image_gray.astype(np.float32)
        mu = cv2.blur(img_float, (window_size, window_size))
        mu_sq = cv2.blur(img_float**2, (window_size, window_size))
        sigma = np.sqrt(np.maximum(mu_sq - mu**2, 0))
        return sigma

def calculate_global_entropy(image_gray):
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.nan_to_num(np.log2(hist + 1e-7))
    return -np.sum(hist * logs)

def process_single_image(img_array, label="Subject"):
    img_native = cv2.imdecode(np.frombuffer(img_array, np.uint8), cv2.IMREAD_COLOR)
    
    # Pillar 1: ICPC (CRITICAL: Must be done on native/cropped resolution, NOT resized)
    # We take a 1024x1024 crop from the center to preserve Bayer filter alignments
    h, w = img_native.shape[:2]
    cy, cx = h // 2, w // 2
    y1, y2 = max(0, cy - 512), min(h, cy + 512)
    x1, x2 = max(0, cx - 512), min(w, cx + 512)
    img_icpc_crop = img_native[y1:y2, x1:x2]
    
    r_diffs = calculate_icpc(img_icpc_crop)
    
    # Now resize for the rest of the visual pillars
    img = cv2.resize(img_native, (512, 512))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    entropy_map = fast_local_entropy(img_gray)
    global_entropy = calculate_global_entropy(img_gray)
    
    # ... (rest of the logic)
    bins = 100
    hist_rg, _ = np.histogram(r_diffs[0].ravel(), bins=bins, range=(-np.pi, np.pi))
    hist_rb, _ = np.histogram(r_diffs[1].ravel(), bins=bins, range=(-np.pi, np.pi))
    hist_gb, _ = np.histogram(r_diffs[2].ravel(), bins=bins, range=(-np.pi, np.pi))
    
    hist_combined = hist_rg + hist_rb + hist_gb
    center_idx = bins // 2
    peak_value = np.sum(hist_combined[center_idx-3:center_idx+4])
    
    # Calculate Peak-to-Median Ratio (PMR)
    background_mask = np.ones(bins, dtype=bool)
    background_mask[center_idx-3:center_idx+4] = False
    median_background = np.median(hist_combined[background_mask])
    
    pmr = peak_value / (median_background + 1e-5)
    
    # Pillar 2: High-Frequency Noise Residual (PRNU proxy)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    noise_variance = np.var(laplacian)
    laplacian_vis = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Pillar 4: Error Level Analysis (ELA)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    ela_diff = cv2.absdiff(img, decimg)
    max_ela = np.max(ela_diff)
    scale = 255.0 / max_ela if max_ela > 0 else 1.0
    ela_image = cv2.convertScaleAbs(ela_diff, alpha=scale)
    ela_score = np.mean(ela_image)
    
    # Pillar 5: Spectral Decay (still calculated as supporting forensic evidence)
    img_gray_native = cv2.cvtColor(img_native, cv2.COLOR_BGR2GRAY)
    ps_prof = azimuthal_average(img_gray_native)
    freq_end = len(ps_prof) - 10
    if freq_end < 20: freq_end = 20
    frequencies = np.arange(10, freq_end)
    ps_log = np.log10(ps_prof[10:freq_end] + 1e-7)
    freqs_log = np.log10(frequencies)
    slope, intercept = np.polyfit(freqs_log, ps_log, 1)

    # --- PRIMARY VERDICT ENGINE: CLIP Few-Shot Classifier ---
    pil_img_native = PILImage.fromarray(cv2.cvtColor(img_native, cv2.COLOR_BGR2RGB))
    
    if clip_classifier is not None and clip_model is not None:
        try:
            clip_feat = get_clip_features(pil_img_native)
            if clip_feat is not None:
                pred = clip_classifier.predict([clip_feat])[0]       # 1=real, 0=fake
                prob = clip_classifier.predict_proba([clip_feat])[0] # [prob_fake, prob_real]
                is_authentic = bool(pred == 1)
                confidence = prob[1] if is_authentic else prob[0]
                verdict = "Authentic Camera Capture" if is_authentic else "AI Deepfake"
                explanation = f"GlassEngine CLIP Forensics: {confidence*100:.1f}% confidence → {verdict}. "
                explanation += f"[Supporting metrics] Spectral Slope: {slope:.2f} | Noise Var: {noise_variance:.1f} | ELA: {ela_score:.1f}"
            else:
                raise ValueError("CLIP feature extraction returned None")
        except Exception as e:
            is_authentic = slope <= -2.2
            verdict = "Authentic Camera Capture" if is_authentic else "AI Deepfake"
            explanation = f"[CLIP error: {e}] Spectral fallback → Slope: {slope:.2f}."
    elif clip_model is not None:
        # CLIP loaded but classifier not trained yet — give instructions
        is_authentic = False
        verdict = "⚠ Classifier Not Trained"
        explanation = "CLIP encoder is ready but no classifier found. Run clip_trainer.py to train on your curated dataset!"
    else:
        # Full fallback: spectral decay
        THRESHOLD_SLOPE = -2.2
        is_authentic = slope <= THRESHOLD_SLOPE
        verdict = "Authentic Camera Capture" if is_authentic else "AI Deepfake"
        explanation = f"[Spectral Fallback] Slope: {slope:.2f}. Noise: {noise_variance:.1f}. ELA: {ela_score:.1f}."

                 
    def fig_to_b64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64

    # Graph 1: Power Spectrum Decay (Spectral Frequency DNA)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(frequencies, ps_prof[10:freq_end], color='cyan', alpha=0.9, label='1D Power Spectrum')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.plot(frequencies, 10**(intercept + slope * freqs_log), 'r--', label=f'Fit Slope: {slope:.2f}')
    ax1.set_title(f"Spectral Frequency DNA")
    ax1.set_xlabel("Spatial Frequency")
    ax1.set_ylabel("Power")
    ax1.legend()
    plt.tight_layout()
    spectral_b64 = fig_to_b64(fig1)
    
    # Graph 2: Entropy
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    im = ax2.imshow(entropy_map, cmap='inferno')
    ax2.set_title(f"Entropy Heatmap ({label})")
    ax2.axis('off')
    fig2.colorbar(im, ax=ax2)
    plt.tight_layout()
    entropy_b64 = fig_to_b64(fig2)
    
    # Graph 3: Noise Residuals
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    im3 = ax3.imshow(laplacian_vis, cmap='gray')
    ax3.set_title(f"Sensor/Noise Residuals ({label})")
    ax3.axis('off')
    plt.tight_layout()
    noise_b64 = fig_to_b64(fig3)
    
    # Graph 4: Error Level Analysis (ELA)
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    im4 = ax4.imshow(cv2.cvtColor(ela_image, cv2.COLOR_BGR2RGB))
    ax4.set_title(f"Error Level Analysis - Diff Map")
    ax4.axis('off')
    plt.tight_layout()
    ela_b64 = fig_to_b64(fig4)
    
    # Graph 5: Inter-Channel Phase Correlation (Pillar 1)
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.bar(np.linspace(-np.pi, np.pi, bins), hist_rg, alpha=0.3, label='R-G Phase Diff', color='red')
    ax5.bar(np.linspace(-np.pi, np.pi, bins), hist_rb, alpha=0.3, label='R-B Phase Diff', color='green')
    ax5.bar(np.linspace(-np.pi, np.pi, bins), hist_gb, alpha=0.3, label='G-B Phase Diff', color='blue')
    ax5.set_title(f"Inter-Channel Phase Correlation (PMR: {pmr:.2f})")
    ax5.set_xlabel("Phase Difference (Radians)")
    ax5.set_ylabel("Count")
    ax5.legend()
    plt.tight_layout()
    icpc_b64 = fig_to_b64(fig5)
    
    return {
        "verdict": verdict,
        "score": float(slope),
        "entropy": float(global_entropy),
        "ratio": float(pmr), # Standardize: ratio is PMR now
        "noise_variance": float(noise_variance),
        "ela_score": float(ela_score),
        "explanation": explanation,
        "spectral_graph": f"data:image/png;base64,{spectral_b64}",
        "icpc_graph": f"data:image/png;base64,{icpc_b64}",
        "entropy_map": f"data:image/png;base64,{entropy_b64}",
        "noise_map": f"data:image/png;base64,{noise_b64}",
        "ela_map": f"data:image/png;base64,{ela_b64}"
    }

@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    img_data = await image.read()
    res = process_single_image(img_data, "Analyzed Subject")
    return res

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
