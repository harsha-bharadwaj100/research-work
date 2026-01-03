import numpy as np
import matplotlib.pyplot as plt
import cv2


def calculate_entropy(image):
    """Calculates Shannon Entropy of an image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.nan_to_num(np.log2(hist + 1e-7))
    return -np.sum(hist * logs)


def get_spectral_magnitude(image):
    """Computes the 2D-FFT Magnitude Spectrum."""
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    return magnitude_spectrum


# 1. Create a 'Natural' Image (Smooth Gradients)
size = 256
x = np.linspace(0, 1, size)
y = np.linspace(0, 1, size)
X, Y = np.meshgrid(x, y)
natural_img = (np.sin(X * 2) * np.cos(Y * 2) * 255).astype(np.uint8)

# 2. Create a 'Synthetically Tampered' Image (Adding Periodic Artifacts/Noise)
# We add a subtle high-frequency checkerboard (common in AI upsampling)
checker = (np.indices((size, size)).sum(axis=0) % 2).astype(np.float32)
tampered_img = cv2.addWeighted(
    natural_img.astype(np.float32), 0.9, checker * 255, 0.1, 0
)
tampered_img = tampered_img.astype(np.uint8)

# 3. Analyze
entropy_nat = calculate_entropy(natural_img)
entropy_tam = calculate_entropy(tampered_img)

spec_nat = get_spectral_magnitude(natural_img)
spec_tam = get_spectral_magnitude(tampered_img)

# 4. Visualization
plt.figure(figsize=(12, 8))

# Natural Analysis
plt.subplot(2, 2, 1)
plt.imshow(natural_img, cmap="gray")
plt.title(f"Natural Image\nEntropy: {entropy_nat:.4f}")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(spec_nat, cmap="magma")
plt.title("Spectral Signature (Natural)\nSmooth, Centralized Energy")
plt.axis("off")

# Tampered Analysis
plt.subplot(2, 2, 3)
plt.imshow(tampered_img, cmap="gray")
plt.title(f"Tampered/AI Image\nEntropy: {entropy_tam:.4f}")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(spec_tam, cmap="magma")
plt.title("Spectral Signature (Tampered)\nNote the Grid/Star Artifacts!")
plt.axis("off")

plt.tight_layout()
plt.savefig("forensic_visualization.png")
print("Visualization saved as 'forensic_visualization.png'")
plt.show()
