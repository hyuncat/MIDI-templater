import numpy as np
from numba import njit
from app.config import AppConfig

class Yin:

    def autocorrelation_fft(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast autocorrelation function implementation using Wiener-Khinchin theorem,
        which computes autocorrelation as the inverse FFT of the signal's power spectrum.

        Step 1 of Yin algorithm, corresponding to equation (1) in Cheveigne, Kawahara 2002.

        Args:
            audio_frame: The current frame of audio samples in Yin algorithm
            tau_max: Check for all time lags up to this value for in autocorrelation

        Returns:
            autocorrelation: The similarity curve.
            power_spec: The power spectrum of the frame.
            amplitudes: Amplitudes of the frame.
        """
        w = audio_frame.size
        tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

        # Zero-pad the audio signal array by the minimum power of 2 which
        # is larger than the window size + tau_max.
        # (circular instead of linear convolution, avoids errors)
        min_fft_size = w + tau_max  # (pad by >tau_max for frame end)

        p2 = (min_fft_size // 32).bit_length()
        nice_fft_sizes = (16, 18, 20, 24, 25, 27, 30, 32)
        size_pad = min(size * (2 ** p2) for size in nice_fft_sizes if size * 2 ** p2 >= min_fft_size)

        # Decompose the signal into its frequency components
        fft_frame = np.fft.rfft(audio_frame, size_pad)  # Use only real part of the FFT (faster)

        # Compute the autocorrelation using Wiener-Khinchin theorem
        power_spectrum = fft_frame * fft_frame.conjugate()
        autocorrelation = np.fft.irfft(power_spectrum)[:tau_max]

        amplitudes = np.abs(fft_frame)

        # Only return valid overlapping values up to window_size-tau_max
        # (type II autocorrelation)
        return autocorrelation[:w-tau_max], power_spectrum, amplitudes
    
    def difference_function(audio_frame: np.ndarray, tau_max: int):
        """
        The square difference function implemented in the seminal YIN paper
        """
        x = np.array(audio_frame, np.float64)  # Ensure float64 precision
        w = x.size
        tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

        autocorr, power_spec, amplitudes = Yin.autocorrelation_fft(x, tau_max)
        
        # Compute m'(tau) - terminology from McLeod
        m_0 = 2*np.sum(x ** 2) # initial m'(0)

        # Compute m'(tau) for each possible tau (McLeod 3.3.4)
        m_primes = np.zeros(tau_max)
        m_primes[0] = m_0
        for tau in range(1, tau_max):
            m_primes[tau] = m_primes[tau-1] - x[tau-1]**2 + x[w-tau]**2

        # Slice m_primes to only contain valid overlapping values
        m_primes = m_primes[:w-tau_max]

        # Compute the square difference function
        sdf = m_primes - 2*autocorr
        return sdf, power_spec, amplitudes
    
    def cmndf(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cumulative Mean Normalized Difference Function (CMNDF).

        The idea is to normalize each d_t(tau) value for all lags based on the mean of the
        cumulative sum of all differences leading up to that point. YIN solution to not
        picking the zero-lag peak.

        Args:
            audio_frame: The current frame of audio samples in Yin algorithm
            tau_max: Check for all time lags up to this value for in autocorr

        Returns:
            cmndf: Array of values where index=tau and value=CMNDF(tau)
            power_spec: The power spectrum of the audio frame
            amplitudes: Amplitudes of the frame
        """
        x = np.array(audio_frame, np.float64)  # Ensure float64 precision
        w = x.size # window size
        tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

        # Compute the difference function
        # autocorr, _ = autocorrelation_fft(x, tau_max)
        # x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
        # diff_fct = x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * autocorr
        diff_fct, power_spec, amplitudes = Yin.difference_function(x, tau_max)

        # Compute the Cumulative Mean Normalized Difference Function (CMNDF)
        cmndf = np.zeros(tau_max)
        cmndf[0] = 1  
        cumulative_diff = 0.0

        for tau in range(1, tau_max):
            cumulative_diff += diff_fct[tau]
            cmndf[tau] = diff_fct[tau] / (cumulative_diff/tau)

        return cmndf, power_spec, amplitudes
    
    @njit
    def parabolic_interpolation(cmndf_frame: np.ndarray, trough_index: int) -> tuple[float, float]:
        """
        Perform parabolic interpolation around a minimum point using Numba for optimization.
        
        Args:
            cmndf_frame: A 1D array of y-values (e.g., CMNDF values).
            trough_index: The index of the minimum point in cmndf_frame.

        Returns:
            A tuple with the interpolated x & y coordinates of the minimum.
        """
        x = trough_index
        if x <= 0 or x >= len(cmndf_frame) - 1:
            return float(x), cmndf_frame[x]  # No interpolation possible at the boundaries
        
        alpha = cmndf_frame[x - 1]
        beta = cmndf_frame[x]
        gamma = cmndf_frame[x + 1]

        denominator = 2 * (alpha - 2 * beta + gamma)
        if denominator == 0:
            return float(x), beta
        
        x_interpolated = x + (alpha - gamma) / denominator
        y_interpolated = beta - (alpha - gamma) * (alpha - gamma) / (4 * denominator)

        return x_interpolated, y_interpolated