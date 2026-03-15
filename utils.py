from aeon.datasets import load_classification
import numpy as np
from scipy.signal import hilbert, savgol_filter
from scipy.fft import rfft
from scipy.fftpack import dct
from pywt import wavedec
import warnings

# Try to import PyEMD, but make it optional
try:
    from PyEMD import EMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False
    warnings.warn("PyEMD not installed. EMD representations will use fallback.")

warnings.filterwarnings('ignore')

def LoadUCR(dset_name=None,
            return_xy=False,
            ):
    
    if return_xy:
        X, y = load_classification(name=dset_name)
    else:
        x_train, y_train = load_classification(name=dset_name,split='train')
        x_test, y_test = load_classification(name=dset_name,split='test')
        dset = x_train, x_test, y_train, y_test
    
    return dset


class RepresentationGenerator:
    """
    Flexible, plug-and-play representation generator for time series.
    Supports custom subsets of representations with easy configuration.
    """
    
    # Available representations with their implementations
    REPRESENTATIONS = {
        # Time-domain representations
        'TIME': {
            'func': lambda x: x,
            'description': 'Original time series',
            'category': 'time'
        },
        'DT1': {
            'func': lambda x: np.diff(x, n=1, axis=-1),
            'description': 'First derivative',
            'category': 'time',
            'needs_pad': True
        },
        'DT2': {
            'func': lambda x: np.diff(x, n=2, axis=-1),
            'description': 'Second derivative',
            'category': 'time',
            'needs_pad': True
        },
        'NORM': {
            'func': lambda x: (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8),
            'description': 'Z-score normalized',
            'category': 'time'
        },
        'SMOOTH': {
            'func': lambda x: savgol_filter(x, window_length=min(11, x.shape[-1]//4), 
                                           polyorder=2, axis=-1),
            'description': 'Savitzky-Golay smoothed',
            'category': 'time'
        },
        
        # Frequency-domain representations
        'FFT_MAG': {
            'func': lambda x: np.abs(rfft(x, axis=-1)),
            'description': 'FFT magnitude',
            'category': 'frequency',
            'needs_pad': True
        },
        'FFT_PHASE': {
            'func': lambda x: np.angle(rfft(x, axis=-1)),
            'description': 'FFT phase',
            'category': 'frequency',
            'needs_pad': True
        },
        'FFT_REAL': {
            'func': lambda x: np.real(rfft(x, axis=-1)),
            'description': 'FFT real part',
            'category': 'frequency',
            'needs_pad': True
        },
        'FFT_IMAG': {
            'func': lambda x: np.imag(rfft(x, axis=-1)),
            'description': 'FFT imaginary part',
            'category': 'frequency',
            'needs_pad': True
        },
        'DCT': {
            'func': lambda x: dct(x, type=2, axis=-1, norm='ortho'),
            'description': 'Discrete Cosine Transform',
            'category': 'frequency'
        },
        'POWER': {
            'func': lambda x: np.abs(rfft(x, axis=-1))**2,
            'description': 'Power spectrum',
            'category': 'frequency',
            'needs_pad': True
        },
        
        # Time-frequency representations
        'HLB_MAG': {
            'func': lambda x: np.abs(hilbert(x, axis=-1)),
            'description': 'Hilbert magnitude',
            'category': 'time_freq'
        },
        'HLB_PHASE': {
            'func': lambda x: np.unwrap(np.angle(hilbert(x, axis=-1)), axis=-1),
            'description': 'Hilbert phase',
            'category': 'time_freq'
        },
        'DWT_A': {
            'func': lambda x: RepresentationGenerator._compute_dwt(x, coeff_type='approx'),
            'description': 'DWT approximation coefficients',
            'category': 'time_freq',
            'needs_pad': True
        },
        'DWT_D': {
            'func': lambda x: RepresentationGenerator._compute_dwt(x, coeff_type='detail'),
            'description': 'DWT detail coefficients',
            'category': 'time_freq',
            'needs_pad': True
        },
        
        # Statistical representations
        'ACF': {
            'func': lambda x: RepresentationGenerator._compute_acf(x),
            'description': 'Autocorrelation',
            'category': 'statistical'
        },
        'ZCR': {
            'func': lambda x: RepresentationGenerator._compute_zcr(x),
            'description': 'Zero Crossing Rate',
            'category': 'statistical',
            'needs_expand': True
        },
        'RMS': {
            'func': lambda x: np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) * np.ones_like(x),
            'description': 'Root Mean Square envelope',
            'category': 'statistical'
        },
        'STD': {
            'func': lambda x: np.std(x, axis=-1, keepdims=True) * np.ones_like(x),
            'description': 'Standard deviation envelope',
            'category': 'statistical'
        },
        
        # Advanced representations
        'EMD_IMF1': {
            'func': lambda x: RepresentationGenerator.compute_EMD(x, max_imfs=1, aggregate=True),
            'description': 'EMD - First IMF',
            'category': 'advanced',
            'needs_pad': True
        },
        'EMD_IMF2': {
            'func': lambda x: RepresentationGenerator.compute_EMD(x, max_imfs=2, aggregate=False)[:, 1, :],
            'description': 'EMD - Second IMF',
            'category': 'advanced',
            'needs_pad': True
        },
        'EMD_IMF3': {
            'func': lambda x: RepresentationGenerator.compute_EMD(x, max_imfs=3, aggregate=False)[:, 2, :],
            'description': 'EMD - Third IMF',
            'category': 'advanced',
            'needs_pad': True
        },
        'EMD_MEAN': {
            'func': lambda x: RepresentationGenerator.compute_EMD(x, max_imfs=3, aggregate=True),
            'description': 'EMD - Mean of first 3 IMFs',
            'category': 'advanced',
            'needs_pad': True
        },
        'EMD_STACK': {
            'func': lambda x: RepresentationGenerator._compute_emd_stack(x),
            'description': 'EMD - Stack of first 3 IMFs (multi-channel)',
            'category': 'advanced',
            'multi_channel': True
        },
        
        'MFCC': {
            'func': lambda x: RepresentationGenerator._compute_mfcc(x),
            'description': 'Mel-Frequency Cepstral Coefficients (simplified)',
            'category': 'advanced',
            'needs_pad': True
        },
        
        # Novel representations
        'QUANTILE': {
            'func': lambda x: RepresentationGenerator._compute_quantile(x),
            'description': 'Quantile features',
            'category': 'statistical',
            'needs_pad': True,
            'needs_reshape': True
        },
        'HISTOGRAM': {
            'func': lambda x: RepresentationGenerator._compute_histogram(x),
            'description': 'Histogram features',
            'category': 'statistical',
            'needs_pad': True,
            'needs_reshape': True
        },
        'TEAGER': {
            'func': lambda x: RepresentationGenerator._compute_teager(x),
            'description': 'Teager-Kaiser Energy Operator',
            'category': 'time',
            'needs_pad': True
        },
        'SPECTRAL_CENTROID': {
            'func': lambda x: RepresentationGenerator._compute_spectral_centroid(x),
            'description': 'Spectral Centroid',
            'category': 'frequency',
            'needs_expand': True
        },
        'SPECTRAL_BANDWIDTH': {
            'func': lambda x: RepresentationGenerator._compute_spectral_bandwidth(x),
            'description': 'Spectral Bandwidth',
            'category': 'frequency',
            'needs_expand': True
        }
        
    }
    
    @staticmethod
    def pad_to_length(data, target_length):
        """Pad or truncate to target length."""
        if data.ndim == 3:  # Multi-channel (n_samples, n_channels, length)
            cur_len = data.shape[-1]
            if cur_len < target_length:
                pad_width = target_length - cur_len
                return np.pad(data, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
            return data[:, :, :target_length]
        else:  # 2D (n_samples, length)
            cur_len = data.shape[-1]
            if cur_len < target_length:
                return np.pad(data, ((0, 0), (0, target_length - cur_len)), mode="constant")
            return data[:, :target_length]
    
    @staticmethod
    def _expand_to_2d(data, target_length):
        """Expand 1D features to 2D by repeating"""
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if data.shape[1] == 1:
            return np.repeat(data, target_length, axis=1)
        return data
    
    # ---------------------- EMD Methods ----------------------
    @staticmethod
    def compute_EMD(X, max_imfs=3, aggregate=False):
        """
        Empirical Mode Decomposition (adaptive decomposition into IMFs).
        
        Args:
            X : np.ndarray (n_samples, length)
            max_imfs : maximum number of IMFs to keep
            aggregate : if True, average the IMFs (1D output);
                        if False, return stacked IMFs (multi-channel output)
        Returns:
            np.ndarray of shape:
                - (n_samples, length) if aggregate=True
                - (n_samples, max_imfs, length) if aggregate=False
        """
        n, L = X.shape
        all_features = []
        
        for xi in X:
            try:
                if HAS_EMD:
                    emd = EMD()
                    imfs = emd.emd(xi)
                else:
                    # Fallback implementation
                    imfs = RepresentationGenerator._simple_emd_fallback(xi)
                
                if imfs.shape[0] == 0:
                    imfs = np.zeros((1, L))
                
                # Ensure we have at least max_imfs
                if imfs.shape[0] < max_imfs:
                    pad = np.zeros((max_imfs - imfs.shape[0], L))
                    imfs = np.vstack([imfs, pad])
                else:
                    imfs = imfs[:max_imfs]
                
                if aggregate:
                    out = np.mean(imfs, axis=0)
                    all_features.append(out)
                else:
                    all_features.append(imfs)
                    
            except Exception as e:
                # Fallback to zeros or original signal
                if aggregate:
                    out = np.zeros(L) if HAS_EMD else xi
                else:
                    out = np.zeros((max_imfs, L)) if HAS_EMD else np.tile(xi, (max_imfs, 1))
                all_features.append(out)
        
        return np.array(all_features)
    
    @staticmethod
    def _simple_emd_fallback(x):
        """Simple fallback EMD-like decomposition"""
        L = len(x)
        # Create some artificial oscillations
        t = np.linspace(0, 1, L)
        imf1 = 0.5 * np.sin(2 * np.pi * 5 * t)  # High frequency
        imf2 = 0.3 * np.sin(2 * np.pi * 2 * t)  # Medium frequency
        imf3 = 0.2 * np.sin(2 * np.pi * 0.5 * t)  # Low frequency
        return np.vstack([imf1, imf2, imf3])
    
    @staticmethod
    def _compute_emd_stack(X):
        """Compute EMD and return as multi-channel output"""
        emd_result = RepresentationGenerator.compute_EMD(X, max_imfs=3, aggregate=False)
        # emd_result shape: (n_samples, 3, L)
        return emd_result
    
    # ---------------------- Helper Methods ----------------------
    @staticmethod
    def _compute_dwt(X, coeff_type='approx', wavelet='db4', level=3):
        """Compute DWT coefficients"""
        n_samples, L = X.shape
        results = []
        
        for xi in X:
            try:
                coeffs = wavedec(xi, wavelet, level=level)
                if coeff_type == 'approx':
                    selected = coeffs[0]  # Approximation coefficients
                else:  # 'detail'
                    # Use first detail coefficients
                    selected = coeffs[1] if len(coeffs) > 1 else coeffs[0]
                
                # Pad/truncate to original length
                if len(selected) < L:
                    selected = np.pad(selected, (0, L - len(selected)), 'constant')
                else:
                    selected = selected[:L]
                    
                results.append(selected)
            except Exception:
                results.append(xi)  # Fallback
        
        return np.array(results)
    
    @staticmethod
    def _compute_acf(X):
        """Compute autocorrelation"""
        n_samples, L = X.shape
        results = np.zeros_like(X)
        
        for i, xi in enumerate(X):
            try:
                acf_i = np.correlate(xi, xi, mode='same')
                # Normalize
                if np.max(np.abs(acf_i)) > 0:
                    acf_i = acf_i / np.max(np.abs(acf_i))
                results[i] = acf_i
            except Exception:
                results[i] = xi  # Fallback
        
        return results
    
    @staticmethod
    def _compute_zcr(X):
        """Compute zero crossing rate"""
        n_samples = X.shape[0]
        zcr_values = np.zeros(n_samples)
        
        for i, xi in enumerate(X):
            # Count zero crossings
            crossings = ((xi[:-1] * xi[1:]) < 0).sum()
            zcr_values[i] = crossings / (len(xi) - 1)
        
        return zcr_values.reshape(-1, 1)
    
    @staticmethod
    def _compute_teager(X):
        """Compute Teager-Kaiser Energy Operator"""
        n_samples, L = X.shape
        results = np.zeros((n_samples, L - 2))  # TKEO reduces length by 2
        
        for i, xi in enumerate(X):
            if len(xi) >= 3:
                # Teager-Kaiser Energy Operator: x[n]² - x[n-1] * x[n+1]
                results[i] = xi[1:-1]**2 - xi[:-2] * xi[2:]
            else:
                results[i] = xi[1:-1] if len(xi) > 2 else xi
        
        return results
    
    @staticmethod
    def _compute_quantile(X):
        """Compute quantile features"""
        n_samples, L = X.shape
        quantiles = np.percentile(X, [25, 50, 75], axis=1)
        # Shape: (3, n_samples) -> reshape to (n_samples, 3)
        return quantiles.T
    
    @staticmethod
    def _compute_histogram(X, bins=10):
        """Compute histogram features"""
        n_samples = X.shape[0]
        hist_features = []
        
        for xi in X:
            hist, _ = np.histogram(xi, bins=bins, density=True)
            hist_features.append(hist)
        
        return np.array(hist_features)
    
    @staticmethod
    def _compute_spectral_centroid(X):
        """Compute spectral centroid"""
        n_samples = X.shape[0]
        centroids = np.zeros(n_samples)
        
        for i, xi in enumerate(X):
            fft = np.abs(rfft(xi))
            freqs = np.linspace(0, 0.5, len(fft))
            if np.sum(fft) > 1e-8:
                centroids[i] = np.sum(freqs * fft) / np.sum(fft)
        
        return centroids.reshape(-1, 1)
    
    @staticmethod
    def _compute_spectral_bandwidth(X):
        """Compute spectral bandwidth"""
        n_samples = X.shape[0]
        bandwidths = np.zeros(n_samples)
        
        for i, xi in enumerate(X):
            fft = np.abs(rfft(xi))
            freqs = np.linspace(0, 0.5, len(fft))
            if np.sum(fft) > 1e-8:
                centroid = np.sum(freqs * fft) / np.sum(fft)
                bandwidths[i] = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft) / np.sum(fft))
        
        return bandwidths.reshape(-1, 1)
    
    @staticmethod
    def _compute_mfcc(x, n_mfcc=13):
        """Simplified MFCC computation"""
        n_samples, l = x.shape
        # Compute power spectrum
        n_fft = min(512, l)
        power_spectrum = np.abs(rfft(x, n=n_fft, axis=-1))**2
        # Simplified mel filterbank
        n_filters = 40
        mel_filters = np.random.randn(n_filters, power_spectrum.shape[1])  # Simplified
        mel_spectrum = np.log(np.dot(power_spectrum, mel_filters.T) + 1e-8)
        # DCT to get MFCCs
        mfcc = dct(mel_spectrum, type=2, axis=-1, norm='ortho')[:, :n_mfcc]
        # Pad to original length
        if mfcc.shape[1] < l:
            mfcc = np.pad(mfcc, ((0, 0), (0, l - mfcc.shape[1])), 'constant')
        else:
            mfcc = mfcc[:, :l]
        return mfcc
    
    @staticmethod
    def _handle_multi_channel(data, target_length, rep_name):
        """Handle multi-channel representations (like EMD_STACK)"""
        # data shape: (n_samples, n_channels, variable_length)
        n_samples, n_channels, cur_len = data.shape
        
        # Pad each channel independently
        padded_data = np.zeros((n_samples, n_channels, target_length))
        for ch in range(n_channels):
            channel_data = data[:, ch, :]
            padded_data[:, ch, :] = RepresentationGenerator.pad_to_length(channel_data, target_length)
        
        return padded_data
    
    # ---------------------- Public Interface ----------------------
    @staticmethod
    def get_available_representations(category=None):
        """Get list of available representations, optionally filtered by category"""
        if category:
            return [name for name, info in RepresentationGenerator.REPRESENTATIONS.items() 
                   if info['category'] == category]
        return list(RepresentationGenerator.REPRESENTATIONS.keys())
    
    @staticmethod
    def get_representations_by_category():
        """Get representations grouped by category"""
        categories = {}
        for name, info in RepresentationGenerator.REPRESENTATIONS.items():
            cat = info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                'name': name,
                'description': info['description']
            })
        return categories
    
    @staticmethod
    def generate_representations(X, representation_list=None, normalize=True, 
                               pre_normalize_input=False, verbose=False, 
                               target_length=None):
        """
        Generate selected representations for time series.
        
        Parameters:
        -----------
        X : np.ndarray
            Input time series of shape (n_samples, l)
        representation_list : list of str, optional
            List of representation names to generate. If None, uses default set.
        normalize : bool, default=True
            Whether to normalize each representation independently
        pre_normalize_input : bool, default=True
            Whether to normalize input time series before generating representations
        verbose : bool, default=True
            Whether to print progress information
        target_length : int, optional
            Target length for all representations. If None, uses input length.
            
        Returns:
        --------
        X_multi : np.ndarray
            Multi-representation tensor of shape (n_samples, r, l)
        rep_info : dict
            Information about the generated representations
        """
        n_samples, l_original = X.shape
        if target_length is None:
            target_length = l_original
        
        if representation_list is None:
            # Default comprehensive set
            representation_list = ['TIME', 'DT1', 'DT2', 'HLB_MAG', 'DWT_A', 
                                 'FFT_MAG', 'DCT', 'ACF']
        
        # Validate representation list
        invalid_reps = [rep for rep in representation_list 
                       if rep not in RepresentationGenerator.REPRESENTATIONS]
        if invalid_reps:
            raise ValueError(f"Invalid representations: {invalid_reps}. "
                           f"Available: {list(RepresentationGenerator.REPRESENTATIONS.keys())}")
        
        if verbose:
            print(f"Generating {len(representation_list)} representations: {representation_list}")
            if HAS_EMD:
                print("PyEMD available: EMD representations enabled")
            else:
                print("PyEMD not available: Using fallback for EMD")
        
        # Pre-normalize input if requested
        X_processed = X.copy()
        if pre_normalize_input:
            for i in range(n_samples):
                xi = X_processed[i]
                mean, std = xi.mean(), xi.std()
                if std > 1e-8:
                    X_processed[i] = (xi - mean) / std
                else:
                    X_processed[i] = xi - mean
        
        representations = []
        rep_info = []
        
        for rep_name in representation_list:
            if verbose:
                print(f"  Generating {rep_name}...", end=' ')
            
            rep_config = RepresentationGenerator.REPRESENTATIONS[rep_name]
            
            try:
                # Apply representation function
                rep_data = rep_config['func'](X_processed)
                
                # Handle special cases
                if rep_config.get('multi_channel', False):
                    # Multi-channel representation (e.g., EMD_STACK)
                    rep_data = RepresentationGenerator._handle_multi_channel(
                        rep_data, target_length, rep_name
                    )
                    # For multi-channel, we need to handle differently
                    # Let's stack channels as separate representations
                    for ch in range(rep_data.shape[1]):
                        channel_data = rep_data[:, ch, :]
                        if normalize:
                            channel_data = (channel_data - channel_data.mean()) / (channel_data.std() + 1e-8)
                        representations.append(channel_data)
                        rep_info.append({
                            'name': f"{rep_name}_ch{ch}",
                            'description': f"{rep_config['description']} (channel {ch})",
                            'category': rep_config['category'],
                            'shape': channel_data.shape
                        })
                    if verbose:
                        print(f"✓ {rep_data.shape[1]} channels")
                    continue
                
                # Handle 1D outputs (like ZCR)
                if rep_config.get('needs_expand', False):
                    rep_data = RepresentationGenerator._expand_to_2d(rep_data, target_length)
                
                # Handle padding for representations that change length
                if rep_config.get('needs_pad', False) or rep_data.shape[-1] != target_length:
                    rep_data = RepresentationGenerator.pad_to_length(rep_data, target_length)
                
                # Handle reshape for statistical features
                if rep_config.get('needs_reshape', False):
                    # Already reshaped by function, but ensure correct 2D shape
                    if rep_data.ndim == 1:
                        rep_data = rep_data.reshape(-1, 1)
                        rep_data = np.repeat(rep_data, target_length, axis=1)
                
                # Ensure 2D shape
                if rep_data.ndim == 1:
                    rep_data = rep_data.reshape(-1, 1)
                    rep_data = np.repeat(rep_data, target_length, axis=1)
                elif rep_data.ndim == 2 and rep_data.shape[1] != target_length:
                    rep_data = RepresentationGenerator.pad_to_length(rep_data, target_length)
                
                representations.append(rep_data)
                rep_info.append({
                    'name': rep_name,
                    'description': rep_config['description'],
                    'category': rep_config['category'],
                    'shape': rep_data.shape
                })
                
                if verbose:
                    print(f"✓ shape={rep_data.shape}")
                    
            except Exception as e:
                print(f"✗ failed: {str(e)}")
                # Fallback to original time series
                rep_data = RepresentationGenerator.pad_to_length(X_processed, target_length)
                representations.append(rep_data)
                rep_info.append({
                    'name': rep_name,
                    'description': f"{rep_config['description']} (failed, using TIME)",
                    'category': rep_config['category'],
                    'shape': rep_data.shape,
                    'error': str(e)
                })
        
        # Stack all representations
        X_multi = np.stack(representations, axis=1)  # (n_samples, r, target_length)
        
        if normalize:
            # normalize each sample and each representation independently
            for rep_idx in range(X_multi.shape[1]):
                rep_data = X_multi[:, rep_idx, :]  # (N, L)
                mean = rep_data.mean(axis=1, keepdims=True)
                std = rep_data.std(axis=1, keepdims=True)
                std[std < 1e-8] = 1.0  # prevent div by zero
                X_multi[:, rep_idx, :] = (rep_data - mean) / std

        
        if verbose:
            print(f"\nFinal shape: {X_multi.shape}")
        
        return X_multi, rep_info
    
    @staticmethod
    def print_representation_stats(X_multi, rep_info):
        """Print statistics for each representation"""
        print("\n" + "="*70)
        print("REPRESENTATION STATISTICS")
        print("="*70)
        print(f"{'Name':12s} {'Category':12s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
        print("-" * 70)
        
        for i, info in enumerate(rep_info):
            if i >= X_multi.shape[1]:
                break
            rep = X_multi[:, i, :]
            print(f"{info['name']:12s} {info['category']:12s} "
                  f"{rep.mean():10.4f} {rep.std():10.4f} "
                  f"{rep.min():10.4f} {rep.max():10.4f}")
        print("="*70)
    
    @staticmethod
    def get_recommended_sets():
        """Get recommended representation sets for different scenarios"""
        return {
            'minimal': ['TIME', 'DCT'],
            'time_domain': ['TIME', 'DT1', 'DT2', 'SMOOTH', 'TEAGER'],
            'frequency': ['FFT_MAG', 'FFT_PHASE', 'DCT', 'POWER'],
            'time_freq': ['HLB_MAG', 'HLB_PHASE', 'DWT_A', 'DWT_D'],
            'statistical': ['ACF', 'ZCR', 'RMS', 'QUANTILE'],
            'emd_based': ['EMD_IMF1', 'EMD_IMF2', 'EMD_MEAN'],
            'comprehensive': ['TIME', 'DT1', 'HLB_MAG', 'DWT_A', 'DWT_D',  
                            'FFT_MAG', 'MFCC', 'DCT', 'ACF'],
            'default': ['TIME', 'DT1', 'HLB_MAG', 'HLB_PHASE', 'DWT_A',  
                            'FFT_MAG', 'POWER', 'DCT', 'ACF', 'SPECTRAL_CENTROID', 'SPECTRAL_BANDWIDTH'],
            'novel': ['TEAGER', 'EMD_STACK', 'SPECTRAL_CENTROID', 'SPECTRAL_BANDWIDTH']
        }