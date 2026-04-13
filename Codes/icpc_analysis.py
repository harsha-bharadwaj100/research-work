import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_phase(channel):
    # 2D FFT
    f_transform = np.fft.fft2(channel)
    f_shift = np.fft.fftshift(f_transform)
    # Extract phase
    phase = np.angle(f_shift)
    return phase

def calculate_icpc(image):
    # Split channels
    b, g, r = cv2.split(image)
    
    # Extract phase for each
    phase_b = extract_phase(b)
    phase_g = extract_phase(g)
    phase_r = extract_phase(r)
    
    # Calculate phase differences
    diff_rg = phase_r - phase_g
    diff_rb = phase_r - phase_b
    diff_gb = phase_g - phase_b
    
    # Wrap to [-pi, pi]
    diff_rg = np.arctan2(np.sin(diff_rg), np.cos(diff_rg))
    diff_rb = np.arctan2(np.sin(diff_rb), np.cos(diff_rb))
    diff_gb = np.arctan2(np.sin(diff_gb), np.cos(diff_gb))
    
    return diff_rg, diff_rb, diff_gb

def fast_local_entropy(image_gray, window_size=5):
    """Fallback if skimage is not available. Simple approximation of local entropy."""
    try:
        from skimage.filters.rank import entropy
        from skimage.morphology import disk
        return entropy(image_gray, disk(window_size))
    except ImportError:
        # Fallback: Local standard deviation as a proxy for information density
        img_float = image_gray.astype(np.float32)
        mu = cv2.blur(img_float, (window_size, window_size))
        mu_sq = cv2.blur(img_float**2, (window_size, window_size))
        sigma = np.sqrt(np.maximum(mu_sq - mu**2, 0))
        return sigma

def process_images(real_path, fake_path, base_name="comparison"):
    print(f"Processing Pair: {base_name}")
    print(f"  Real: {real_path}")
    print(f"  Fake: {fake_path}")
    
    real_img = cv2.imread(real_path)
    fake_img = cv2.imread(fake_path)
    
    if real_img is None or fake_img is None:
        print("Error loading images!")
        return
    
    # Resize for faster processing and normalized comparison
    real_img = cv2.resize(real_img, (512, 512))
    fake_img = cv2.resize(fake_img, (512, 512))
    
    # 1. Calculate ICPC
    r_diffs = calculate_icpc(real_img)
    f_diffs = calculate_icpc(fake_img)
    
    # 2. Calculate Local Entropy Heatmap
    real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    fake_gray = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
    
    real_entropy = fast_local_entropy(real_gray)
    fake_entropy = fast_local_entropy(fake_gray)
    
    # 3. Plotting
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    
    # Row 1: Original Images
    axes[0, 0].imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Authentic Image: {base_name}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Deepfake Image: {base_name}")
    axes[0, 1].axis('off')
    
    # Row 2: ICPC Phase Sync Graphs (Histograms)
    bins = 100
    axes[1, 0].hist(r_diffs[0].ravel(), bins=bins, alpha=0.5, label='R-G', color='y')
    axes[1, 0].hist(r_diffs[1].ravel(), bins=bins, alpha=0.5, label='R-B', color='m')
    axes[1, 0].hist(r_diffs[2].ravel(), bins=bins, alpha=0.5, label='G-B', color='c')
    axes[1, 0].set_title("Phase Sync Graph (Real)\nPhysical Sensors Often Preserve Sync")
    axes[1, 0].set_xlim([-np.pi, np.pi])
    axes[1, 0].legend()
    
    axes[1, 1].hist(f_diffs[0].ravel(), bins=bins, alpha=0.5, label='R-G', color='y')
    axes[1, 1].hist(f_diffs[1].ravel(), bins=bins, alpha=0.5, label='R-B', color='m')
    axes[1, 1].hist(f_diffs[2].ravel(), bins=bins, alpha=0.5, label='G-B', color='c')
    axes[1, 1].set_title("Phase Sync Graph (Fake)\nShattered/Anomalous Phase Details")
    axes[1, 1].set_xlim([-np.pi, np.pi])
    axes[1, 1].legend()
    
    # Row 3: Entropy Heatmaps
    im1 = axes[2, 0].imshow(real_entropy, cmap='inferno')
    axes[2, 0].set_title("Entropy-Based Heatmap (Real)")
    axes[2, 0].axis('off')
    fig.colorbar(im1, ax=axes[2, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[2, 1].imshow(fake_entropy, cmap='inferno')
    axes[2, 1].set_title("Entropy-Based Heatmap (Fake)")
    axes[2, 1].axis('off')
    fig.colorbar(im2, ax=axes[2, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    out_path = os.path.join("Visualization", f"icpc_entropy_{base_name}.png")
    os.makedirs("Visualization", exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    real_img_dir = os.path.join("AI_img_testing", "Real_images")
    fake_img_dir = os.path.join("AI_img_testing", "DF_images")
    
    real_imgs = [f for f in os.listdir(real_img_dir) if f.endswith(('.jpg', '.png'))]
    
    for real_name in real_imgs:
        base_name = os.path.splitext(real_name)[0]
        fake_candidates = [f for f in os.listdir(fake_img_dir) if f.startswith(base_name) and f.endswith(('.jpg', '.png'))]
        if fake_candidates:
            real_path = os.path.join(real_img_dir, real_name)
            fake_path = os.path.join(fake_img_dir, fake_candidates[0])
            process_images(real_path, fake_path, base_name)
        else:
            print(f"Skipping {real_name}: No corresponding fake image found.")
