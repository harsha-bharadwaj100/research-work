import os
import cv2
import numpy as np

def extract_phase(channel):
    f_transform = np.fft.fft2(channel)
    f_shift = np.fft.fftshift(f_transform)
    return np.angle(f_shift)

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

def calculate_global_entropy(image_gray):
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.nan_to_num(np.log2(hist + 1e-7))
    return -np.sum(hist * logs)

def evaluate_image(filepath):
    img = cv2.imread(filepath)
    if img is None: return None
    img = cv2.resize(img, (512, 512))
    
    r_diffs = calculate_icpc(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global_entropy = calculate_global_entropy(img_gray)
    
    bins = 100
    hist_rg, _ = np.histogram(r_diffs[0].ravel(), bins=bins, range=(-np.pi, np.pi))
    hist_rb, _ = np.histogram(r_diffs[1].ravel(), bins=bins, range=(-np.pi, np.pi))
    hist_gb, _ = np.histogram(r_diffs[2].ravel(), bins=bins, range=(-np.pi, np.pi))
    
    center_idx = bins // 2
    peak_score = np.sum(hist_rg[center_idx-3:center_idx+4]) + \
                 np.sum(hist_rb[center_idx-3:center_idx+4]) + \
                 np.sum(hist_gb[center_idx-3:center_idx+4])
                 
    ratio1 = (peak_score / 1000) / (global_entropy + 1e-5)
    ratio2 = peak_score * global_entropy # test inverse
    
    return peak_score, global_entropy, ratio1, ratio2

import json

def run_test():
    real_dir = r"c:\D-sim\pyprojs\research-work\AI_img_testing\Real_images"
    fake_dir = r"c:\D-sim\pyprojs\research-work\AI_img_testing\DF_images"
    
    results = {"REAL": {}, "FAKE": {}}
    
    for label, d in [("REAL", real_dir), ("FAKE", fake_dir)]:
        for f in os.listdir(d):
            if not f.endswith(('.jpg', '.png')): continue
            path = os.path.join(d, f)
            res = evaluate_image(path)
            if res:
                peak, ent, r1, r2 = res
                results[label][f] = {"PEAK": int(peak), "ENTROPY": float(ent), "P/E": float(r1), "P*E": float(r2)}
    
    with open("clean_results.json", "w") as out:
        json.dump(results, out, indent=4)
    print("Done")
if __name__ == "__main__":
    run_test()
