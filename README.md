# 🔍 GlassEngine: Forensic Image Verification System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB?logo=react)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GlassEngine** is a high-fidelity forensic verification system designed to detect AI-generated media (deepfakes) by triangulating physical, mathematical, and neural evidence. Unlike standard detectors that rely on simple CNNs, GlassEngine analyzes the "Digital DNA" of images—mapping hardware physics constraints and latent mathematical artifacts that generative AI cannot yet replicate. 

-----------------------------

## 🛡️ The 5 Forensic Pillars (Evidence Dashboard)

GlassEngine utilizes five distinct layers of physical and mathematical proof to generate a comprehensive integrity verdict:

### 1. High-Frequency Enhanced Inter-Channel Phase Correlation (HF-ICPC)
Measures the physical "Phase Lock" between R, G, and B sensors common in real Bayer-filter hardware.
**[Novelty Update]**: We applied an Unsharp Masking pre-filter before Phase Extraction. This strongly focuses the correlation algorithm on micro-textures and structural high-frequency boundaries—precisely where generative AI models fail to maintain physical rules.
- **Authentic**: Shows a sharp, synchronized spike at 0 phase difference.
- **Deepfake**: Shows a flat or scattered distribution because AI generates color channels independently.

### 2. Rényi Collision Entropy Heatmap
Measures micro-chaos and information density at the pixel level.
**[Novelty Update]**: We replaced the standard Shannon Entropy formula with **Rényi Entropy (Order 2 - Collision Entropy)**. This mathematical modification creates a stricter calculation for probability collisions amongst pixel intensities, heightening the detector's sensitivity to synthetic image generation without skewing the output metrics.
- **Authentic**: The natural world is consistently chaotic/complex at a micro-texture level.
- **Deepfake**: AI math is "efficient"—it creates patches of suspiciously smooth "low-entropy" zones.

### 3. Sensor Noise Residual (PRNU Proxy)
Extracts the "hardware fingerprint" unique to physical camera sensors using Laplacian high-pass filters.
- **Authentic**: Contains a uniform, stochastic hardware grain (Photo Response Non-Uniformity).
- **Deepfake**: "Sterile" or patchy noise grain that fails to follow physical sensor laws.

### 4. Error Level Analysis (ELA)
Maps JPEG recompression differential mismatches across the image frame.
- **Authentic**: Consistent compression levels across the entire frame.
- **Deepfake**: Forged or AI-generated areas "glow" in ELA maps due to disparate compression histories.

### 5. Spectral Frequency DNA
Analyzes the 1D power spectrum decay to identify synthetic artifacts in the frequency domain.
- **Authentic**: Follows natural $1/f^\alpha$ decay patterns.
- **Deepfake**: Exhibits "spectral spikes" or unnatural slopes due to upsampling algorithms.

---

## 🧠 Primary Verdict Engine: CLIP Few-Shot Model

At the heart of GlassEngine is a custom-trained **CLIP Few-Shot Classifier** using the `openai/clip-vit-base-patch32` encoder.

- **Few-Shot Learning**: Trained on a curated dataset of high-fidelity real photos vs. Gemini-generated synthetic images.
- **Neural Forensics**: Identifies subtle perceptual "hallucinations" in textures that are invisible to the human eye but unmistakable in CLIP's high-dimensional feature space.
- **Adaptive Fallback**: In the absence of a trained model, the system automatically falls back to physical spectral analysis to ensure 100% uptime.

---

## 🚀 Tech Stack & Architecture

- **Backend**: FastAPI (Python 3.12+) managed via [uv](https://github.com/astral-sh/uv).
- **Frontend**: React + Vite for a real-time forensic dashboard.
- **Core Libraries**: 
  - `PyTorch` & `Transformers` (CLIP)
  - `OpenCV` & `SciPy` (Signal Processing)
  - `Scikit-Learn` (Logistic Regression Pipeline)
  - `Matplotlib` (Forensic Visualization)

---

## 🛠️ Installation & Setup

### 1. Backend Setup
The backend requires `uv` for high-performance dependency management.

```powershell
# Install dependencies
uv sync

# Run the FastAPI server
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
Navigate to the frontend directory to start the dashboard.

```powershell
cd frontend
npm install
npm run dev
```

### 3. Training the Classifier
If you wish to retrain the CLIP classifier on your own dataset:
```powershell
uv run python clip_trainer.py
```

---

## 📂 Project Structure

```text
research-work/
├── backend/            # FastAPI Application & ML Models
├── frontend/           # React + Vite Dashboard
├── Codes/              # Specialized Forensic Analysis Scripts
├── AI_img_testing/     # Curated Dataset for Few-Shot Training
├── clip_trainer.py     # CLIP Few-Shot Training Script
├── pyproject.toml      # Python Dependencies (UV)
└── README.md           # Project Documentation
```

---

## 📜 Academic Citations (2024–2025)

GlassEngine is built upon foundational research in generative AI forensics:
- **CLIP & Few-Shot**: *Cappelletti et al. (ICPR, 2024)* & *Tan et al. (AAAI, 2025)*.
- **Hardware Forensics**: *Khan et al. (ICCK, 2025)* (ICPC Phase Analysis).
- **Latent Entropy**: *f-InfoED (ArXiv, 2024)*.
- **Sensor Noise**: *NoiseDF (2024)* & *IJSAT Hybrid Forensic Framework (2025)*.

---

*Developed by Harsha Bharadwaj & Guru Raj Patil.*
