def extract_features_np(audio_np, sr=SR):
    y = audio_np.flatten().astype(np.float32)

    if y.size == 0:
        return None

    # Ensure min length to avoid librosa crashes
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    # -----------------------------
    # BASIC FEATURES (3 stats each)
    # -----------------------------
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # take multiple statistics, not just mean
    def stats(x):
        return [np.mean(x), np.std(x), np.median(x)]

    feat_centroid = stats(centroid)
    feat_rms = stats(rms)
    feat_zcr = stats(zcr)
    feat_bandwidth = stats(bandwidth)

    # -----------------------------
    # MFCCs (13 coefficients)
    # -----------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    # -----------------------------
    # Spectral Contrast (7 bands)
    # -----------------------------
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_means = np.mean(contrast, axis=1)

    # -----------------------------
    # Rolloff (shape of spectrum)
    # -----------------------------
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feat_rolloff = stats(rolloff)

    # -----------------------------
    # FINAL FEATURE VECTOR
    # -----------------------------
    final_features = np.concatenate([
        feat_centroid,
        feat_rms,
        feat_zcr,
        feat_bandwidth,
        mfcc_means,
        mfcc_stds,
        contrast_means,
        feat_rolloff
    ])

    return final_features
