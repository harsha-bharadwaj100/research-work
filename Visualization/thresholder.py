import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 1. Simulate the Siamese output for Simple vs Complex matches
# In a real run, you would use: distance = model.predict([img1, img2])
simple_match_dist = 0.05  # Digit '1' vs Digit '1'
complex_match_dist = 0.35  # Digit '8' vs Digit '8'


def get_entropy(img):
    hist, _ = np.histogram(img, bins=256, range=(0, 1))
    return entropy(hist / hist.sum(), base=2)


# 2. Visualize the Threshold Problem
def visualize_threshold_need(img_simple, img_complex):
    ent_s = get_entropy(img_simple)
    ent_c = get_entropy(img_complex)

    # Logic: Lower entropy needs a tighter (smaller) threshold to avoid false matches
    # Higher entropy needs a wider (larger) threshold to allow for natural variance
    suggested_threshold_s = 0.1 * ent_s  # Hypothetical adaptive logic
    suggested_threshold_c = 0.1 * ent_c

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].imshow(img_simple.squeeze(), cmap="gray")
    ax[0].set_title(
        f"Simple (Entropy: {ent_s:.2f})\nMeasured Dist: {simple_match_dist}\nSuggested Max Threshold: {suggested_threshold_s:.2f}"
    )

    ax[1].imshow(img_complex.squeeze(), cmap="gray")
    ax[1].set_title(
        f"Complex (Entropy: {ent_c:.2f})\nMeasured Dist: {complex_match_dist}\nSuggested Max Threshold: {suggested_threshold_c:.2f}"
    )

    plt.show()


# Example: Use a '1' and an '8' from your dataset
# visualize_threshold_need(test_images[index_of_1], test_images[index_of_8])
