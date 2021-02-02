import numpy as np

def wave(sig, shift, centroid_length):
    sample_length = sig.shape[0]
    wave = sig[shift:shift + centroid_length]
    lnan, rnan = shift, sample_length - shift - centroid_length
    wave = np.r_[np.full(lnan, np.nan), wave, np.full(rnan, np.nan)]

    return wave
