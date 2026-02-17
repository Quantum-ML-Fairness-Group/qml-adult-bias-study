import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# PCA embedding
def apply_pca(X_train, X_test, n_qubits):
    print(f"Applying PCA to reduce dimension to {n_qubits}...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_qubits)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Normalize to [-pi, pi]
    X_train_pca = np.clip(X_train_pca, -3, 3) * (np.pi / 3)
    X_test_pca = np.clip(X_test_pca, -3, 3) * (np.pi / 3)

    return X_train_pca, X_test_pca


# Amplitude embedding
def prepare_amplitude_encoding(X_train, X_test):

    n_features = X_train.shape[1]
    print(f"Original feature dimension: {n_features}") # 大概是 100 多

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    
    target_dim = 2 ** np.ceil(np.log2(n_features)).astype(int)
    pad_width = target_dim - n_features
    print(f"Padding to {target_dim} dimensions (requires {int(np.log2(target_dim))} qubits)")

    X_train_pad = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant')
    X_test_pad = np.pad(X_test, ((0, 0), (0, pad_width)), mode='constant')

    norm_train = np.linalg.norm(X_train_pad, axis=1, keepdims=True)
    norm_test = np.linalg.norm(X_test_pad, axis=1, keepdims=True)
    
    norm_train[norm_train == 0] = 1
    norm_test[norm_test == 0] = 1

    X_train_norm = X_train_pad / norm_train
    X_test_norm = X_test_pad / norm_test

    return X_train_norm, X_test_norm, int(np.log2(target_dim))