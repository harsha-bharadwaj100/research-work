# Project GlassEngine: Mathematical Foundations

This document outlines the core algorithms and mathematical formulas used in the GlassEngine forensic analysis system, specifically detailing Pillar 1 (Shannon Entropy) and Pillar 2 (Spectral DNA).

---

## 1. Inter-Channel Phase Correlation (ICPC)

The ICPC metric measures the structural synchronization between color channels (Red, Green, Blue) in the frequency domain. Authentic camera captures maintain a physical phase lock across channels due to the Bayer filter, whereas AI generation renders channels mathematically, destroying this lock.

### Algorithm Steps

1. **Channel Separation:** Extract the individual $R$, $G$, and $B$ channels from the image $I(x,y)$.
2. **2D Fast Fourier Transform (FFT):** Convert each spatial color channel into the frequency domain.
   $$F_c(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I_c(x,y) e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}$$
   *Where $c \in \{R, G, B\}$, $(x,y)$ are spatial coordinates, and $(u,v)$ are frequency coordinates.*

3. **Phase Extraction:** Calculate the angle (phase) of the complex frequency components.
   $$\Phi_c(u,v) = \arctan\left(\frac{\text{Im}(F_c(u,v))}{\text{Re}(F_c(u,v))}\right)$$

4. **Phase Difference Calculation:** Compute the raw elemental phase differences between the channels.
   $$\Delta\Phi_{RG}(u,v) = \Phi_R(u,v) - \Phi_G(u,v)$$
   $$\Delta\Phi_{RB}(u,v) = \Phi_R(u,v) - \Phi_B(u,v)$$
   $$\Delta\Phi_{GB}(u,v) = \Phi_G(u,v) - \Phi_B(u,v)$$

5. **Phase Wrapping:** Wrap the absolute differences to the principal interval $[-\pi, \pi]$ to avoid artificial geometric discontinuities.
   $$\Delta\Phi'_{c_1 c_2} = \text{arctan2}(\sin(\Delta\Phi_{c_1 c_2}), \cos(\Delta\Phi_{c_1 c_2}))$$

6. **Histogram Generation (Phase Sync Graph):** Plot the probability distribution (histogram) of the wrapped phase differences. An authentic image will show an extreme spike at $0$, while a deepfake will show a randomized, flat, or shattered distribution.

---

## 2. Spatial Shannon Entropy Heatmap

Shannon Entropy measures the local information density or structural unpredictability of an image. AI generators optimize for macro-visuals but struggle to recreate the micro-stochastic noise of physical photons, resulting in abnormally low localized entropy.

### Algorithm Steps

1. **Grayscale Conversion:** Convert the RGB image to a single luminance channel $I_{\text{gray}}(x,y)$.
2. **Defining the Local Window:** Define a sliding structural window $W$ (e.g., a $5 \times 5$ pixel disk) centered at each pixel coordinate $(x_c, y_c)$.
3. **Probability Distribution:** For the pixels strictly inside the window $W(x_c, y_c)$, calculate the histogram to find the probability $p_i$ of each possible pixel intensity value $i$ occurring (where $i \in [0, 255]$ for standard 8-bit images).
   $$p_i = \frac{\text{Count of pixels with intensity } i \text{ in } W}{\text{Total number of pixels in } W}$$

4. **Shannon Entropy Formula:** Apply Claude Shannon's 1948 Information Theory formula to calculate the absolute scalar entropy $H$ for that specific window.
   $$H(x_c, y_c) = -\sum_{i=0}^{255} p_i \log_2(p_i)$$
   *Note: In information theory, if $p_i = 0$, the expression $p_i \log_2(p_i)$ is evaluated as $0$ by convention.*

5. **Heatmap Generation:** Assign the computed entropy scalar $H(x_c, y_c)$ to the exact center pixel $(x_c, y_c)$ to construct the final 2D spatial entropy map (Heatmap). Color mapping (e.g., the 'inferno' colormap) is then applied to translate low entropy to dark regions and high entropy to bright regions.
