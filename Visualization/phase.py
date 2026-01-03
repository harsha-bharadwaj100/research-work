import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_spectral_dna(image):
    # Perform 2D FFT
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    # Magnitude Spectrum (What standard tools see)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    # Phase Spectrum (What OmniTrust sees)
    phase_spectrum = np.angle(f_shift)

    return magnitude_spectrum, phase_spectrum


# Generate a "Real" looking gradient and a "Fake" looking grid
real_sim = np.sin(np.linspace(0, 10, 100)) * np.cos(np.linspace(0, 10, 100))[:, None]
fake_sim = np.zeros((100, 100))
fake_sim[::4, ::4] = 1.0  # Unnatural AI-like grid artifacts

mag_r, phase_r = visualize_spectral_dna(real_sim)
mag_f, phase_f = visualize_spectral_dna(fake_sim)

# Plotting the "Hidden DNA"
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].imshow(mag_r, cmap="gray")
ax[0, 0].set_title("Natural Magnitude (Smooth)")
ax[0, 1].imshow(phase_r, cmap="twilight")
ax[0, 1].set_title("Natural Phase (Ordered)")

ax[1, 0].imshow(mag_f, cmap="gray")
ax[1, 0].set_title("AI Artifact Magnitude (Stars/Dots)")
ax[1, 1].imshow(phase_f, cmap="twilight")
ax[1, 1].set_title("AI Artifact Phase (Disordered)")
plt.show()
