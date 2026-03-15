import numpy as np
import torch

from representation_generator import RepresentationGenerator
from utils import LoadUCR, RepresentationGenerator


def normalize_per_sample(X):
    """
    Apply z-normalization per sample (and per channel if multivariate).
    Matches official LITE implementation.
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    X = np.array(X, dtype=np.float32)

    if X.ndim == 2:
        # [N, L]
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        X = (X - means) / stds

    elif X.ndim == 3:
        # [N, C, L]
        for i in range(X.shape[0]):
            for c in range(X.shape[1]):
                std = X[i, c].std()
                if std == 0: std = 1.0
                X[i, c] = (X[i, c] - X[i, c].mean()) / std

    return torch.tensor(X, dtype=torch.float32)


def load_ucr(
    dataset_name,
    use_original_split=False,
    normalize=True,
    representation_list = None,
    print_stats=False
):
    """
    Load UCR dataset and generate multi-representation tensor.

    Parameters
    ----------
    dataset_name : str
        Name of UCR dataset (e.g. 'ECG200')
    use_original_split : bool
        If True, return train/test separately.
        If False, merge and return full dataset (recommended for MC-CV).
    normalize : bool
        Whether to normalize representations.
    print_stats : bool
        Print per-representation statistics.

    Returns
    -------
    If use_original_split:
        Xtr, ytr, Xte, yte
    Else:
        X, y
    """
    
    if representation_list is None:
        representation_list = ['TIME', 'DT1', 'HLB_MAG', 'HLB_PHASE', 'DWT_A',  
                        'FFT_MAG', 'POWER', 'DCT', 'ACF', 'SPECTRAL_CENTROID', 'SPECTRAL_BANDWIDTH']

    try:
        dset, metadata = LoadUCR(
            dataset_name,
            repo="ucr_drive",
            orig_split=True
        )
        x_train, x_test, y_train, y_test = dset
    except Exception as e:
        raise RuntimeError(f"Failed to load UCR dataset {dataset_name}: {e}")

    # Ensure numpy arrays
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test  = np.asarray(x_test,  dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    # UCR labels are often 1-based → convert to 0-based
    if y_train.min() == 1:
        y_train -= 1
        y_test  -= 1

    # -----------------------------------------------------------------
    # Generate representations
    # -----------------------------------------------------------------
    Xtr, _ = RepresentationGenerator.generate_representations(
        x_train, representation_list=representation_list, verbose=False
    )
    Xte, _ = RepresentationGenerator.generate_representations(
        x_test, representation_list=representation_list, verbose=False
    )
    
    # Before: X_train, X_val, X_test are numpy arrays
    # Xtr = normalize_per_sample(Xtr)    
    # Xte = normalize_per_sample(Xte)

    if print_stats:
        print(f"\n[{dataset_name}] TRAIN")
        RepresentationGenerator.get_representation_stats(Xtr)
        print(f"\n[{dataset_name}] TEST")
        RepresentationGenerator.get_representation_stats(Xte)

    if use_original_split:
        return Xtr, y_train, Xte, y_test

    # -----------------------------------------------------------------
    # Merge for Monte-Carlo CV
    # -----------------------------------------------------------------
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    return X, y
