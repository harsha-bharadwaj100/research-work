import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


def calculate_shannon_entropy(image):
    # Calculate histogram of grayscale values
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 1))
    prob_dist = hist / hist.sum()
    # Remove zeros for log calculation
    prob_dist = prob_dist[prob_dist > 0]
    return entropy(prob_dist, base=2)


# Create two dummy images: One simple, one complex
simple_img = np.zeros((28, 28))
simple_img[10:18, 10:18] = 1.0  # A simple square

complex_img = np.random.rand(28, 28)  # Random noise (High information)

# Calculate Entropies
ent_simple = calculate_shannon_entropy(simple_img)
ent_complex = calculate_shannon_entropy(complex_img)

# Visualize
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(simple_img, cmap="gray")
ax[0].set_title(f"Low Entropy: {ent_simple:.2f}\n(Strict Threshold Needed)")
ax[1].imshow(complex_img, cmap="gray")
ax[1].set_title(f"High Entropy: {ent_complex:.2f}\n(Lenient Threshold Needed)")
plt.show()
