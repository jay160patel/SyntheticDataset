from scipy.spatial.distance import jensenshannon
import numpy as np


def compute_js_divergence(real_col, synthetic_col, bins=20):
    real_hist, _ = np.histogram(real_col.dropna(), bins=bins, density=True)
    synthetic_hist, _ = np.histogram(synthetic_col.dropna(), bins=bins, density=True)

    real_hist += 1e-8
    synthetic_hist += 1e-8

    real_hist = real_hist / real_hist.sum()
    synthetic_hist = synthetic_hist / synthetic_hist.sum()

    return jensenshannon(real_hist, synthetic_hist)
