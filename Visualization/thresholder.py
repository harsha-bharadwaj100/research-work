import numpy as np
from scipy.stats import entropy
from PIL import Image
import os

# Load your existing model
# model = tf.keras.models.load_model("siamese_mnist_model.keras", compile=False)


def get_entropy(img):
    # Standard Shannon Entropy calculation for pixel distribution
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 1))
    prob_dist = hist / hist.sum()
    return entropy(prob_dist[prob_dist > 0], base=2)


def load_and_preprocess(path):
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    return img.reshape(28, 28, 1)


# Load images
img1_simple = load_and_preprocess(os.path.join("..", "test_images", "img1_simple.png"))
img2_simple = load_and_preprocess(os.path.join("..", "test_images", "img2_simple.png"))
img1_complex = load_and_preprocess(
    os.path.join("..", "test_images", "img1_complex.jpg")
)
img2_complex = load_and_preprocess(
    os.path.join("..", "test_images", "img2_complex.jpg")
)

# Get Model Distances
# dist_simple = model.predict([img1_simple, img2_simple])[0][0]
# dist_complex = model.predict([img1_complex, img2_complex])[0][0]

# Calculate Entropies
ent_s = get_entropy(img1_simple)
ent_c = get_entropy(img1_complex)

print(f"Simple Pair (1s): Entropy {ent_s:.2f}")
print(f"Complex Pair (8s): Entropy {ent_c:.2f}")
