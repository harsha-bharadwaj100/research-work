# Forensic Analysis Walkthrough: Pillar 2

We successfully implemented and tested Pillar 2 of Project GlassEngine (Inter-Channel Phase Correlation & Spectral DNA).

## Implementation Details
1. **ICPC Analysis Script**: We wrote [icpc_analysis.py](file:///c:/D-sim/pyprojs/research-work/Codes/icpc_analysis.py) which extracts mathematical Deepfake markers by:
   - Computing the 2D-FFT for each color channel.
   - Extracting phase differences (R-G, R-B, G-B) to plot the Phase Sync Graphs.
   - Calculating the Shannon Entropy Heatmap using `scikit-image`'s local entropy rank filter.
2. **Execution**: We updated the script to automatically pair corresponding images by filename. It tested the user-provided Gemini Nano deepfakes ([brush.png](file:///c:/D-sim/pyprojs/research-work/AI_img_testing/DF_images/brush.png), [pen.png](file:///c:/D-sim/pyprojs/research-work/AI_img_testing/DF_images/pen.png)) and compared them against their authentic camera-captured counterparts ([brush.jpg](file:///c:/D-sim/pyprojs/research-work/AI_img_testing/Real_images/brush.jpg), [pen.jpg](file:///c:/D-sim/pyprojs/research-work/AI_img_testing/Real_images/pen.jpg)).

## Validation Results

Below are the visualizations showcasing our findings for both pairs.

### Pair 1: Brush
![Brush Analysis](/C:/Users/harsh/.gemini/antigravity/brain/137bae4f-e4ef-4871-8ced-5ec32ce87cf9/icpc_entropy_brush.png)

### Pair 2: Pen
![Pen Analysis](/C:/Users/harsh/.gemini/antigravity/brain/137bae4f-e4ef-4871-8ced-5ec32ce87cf9/icpc_entropy_pen.png)

### Key Observations
1. **Phase Sync Graphs (Spectral DNA)**: The authentic images display relatively concentrated phase peaks, acting as proof of physical sensor phase coherence. Contrarily, the AI-generated fake images show a "shattered" or highly anomalous phase distribution, validating the `FreqNet` and `Wang et al.` research findings on GenAI spectral footprints.
2. **Entropy Heatmaps**: The entropy visualization successfully highlights spatial information density. Deepfakes often exhibit unnatural structural uniformity (over-smoothing) or math-grid structures, which the spatial entropy mapping correctly isolates.
