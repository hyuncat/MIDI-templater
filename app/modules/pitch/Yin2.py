import numpy as np
import time
from numba import njit
import scipy
import librosa
import scipy.signal as signal
from dataclasses import dataclass
from typing import Optional

@dataclass
class Pitch:
    """
    Stores pitch properties in the CMNDF function.
    """
    time: float
    frequency: float
    probability: float
    pitch_bin: Optional[int] = None

class PYin:
    def __init__(self):
        pass

    def high_pass_irr_filter(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        # High pass filter to lower intensity of low frequency noise
        # Based on McLeod
        nyquist = sample_rate / 2
        CUTOFF_FREQ = 150
        normal_cutoff = CUTOFF_FREQ / nyquist
        sos = signal.iirfilter(
            N=4, Wn=normal_cutoff, rp=3, 
            btype='highpass', 
            ftype='butter', 
            output='sos', 
            fs=sample_rate
        )
        audio_data = signal.sosfilt(sos, audio_data)
        return audio_data

    def autocorrelation_fft(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast autocorrelation function implementation using Wiener-Khinchin theorem,
        which computes autocorrelation as the inverse FFT of the signal's power spectrum.

        Step 1 of Yin algorithm, corresponding to equation (1) in Cheveigne, Kawahara 2002.

        Args:
            audio_frame: The current frame of audio samples in Yin algorithm
            tau_max: The time-lag to check for in the autocorrelation
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

        # Only return valid overlapping values up to window_size-tau_max
        # (type II autocorrelation)
        return autocorrelation[:w-tau_max], power_spectrum


    def square_difference_fct(audio_frame: np.ndarray, tau_max: int):
        """
        The square difference function implemented in the seminal YIN paper.
        """
        x = np.array(audio_frame, np.float64)  # Ensure float64 precision
        w = x.size
        tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

        autocorr, power_spec = PYin.autocorrelation_fft(x, tau_max)
        
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
        return sdf, power_spec


    def cmndf(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Cumulative Mean Normalized Difference Function (CMNDF) implementation.
        """
        x = np.array(audio_frame, np.float64)  # Ensure float64 precision
        w = x.size # window size
        tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

        # Compute the difference function
        # autocorr, _ = autocorrelation_fft(x, tau_max)
        # x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
        # diff_fct = x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * autocorr
        diff_fct, power_spec = PYin.square_difference_fct(x, tau_max)

        # Compute the Cumulative Mean Normalized Difference Function (CMNDF)
        cmndf = np.zeros(tau_max)
        cmndf[0] = 1  
        cumulative_diff = 0.0

        for tau in range(1, tau_max):
            cumulative_diff += diff_fct[tau]
            cmndf[tau] = diff_fct[tau] / (cumulative_diff/tau)

        return cmndf, power_spec


    @njit
    def parabolic_interpolation_numba(cmndf_frame: np.ndarray, trough_index: int) -> tuple[float, float]:
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


    @njit
    def compute_trough_probabilities(trough_prior, trough_thresholds, beta_probs, no_trough_prob):
        """
        Compute the probabilities of each trough using the prior distribution
        and beta distribution of thresholds, optimized with Numba.
        """
        probs = trough_prior.dot(beta_probs)
        global_min = np.argmin(probs)
        n_thresholds_below_min = np.count_nonzero(~trough_thresholds[global_min, :])
        probs[global_min] += no_trough_prob * np.sum(beta_probs[:n_thresholds_below_min])
        return probs


    def pitch_probabilities(cmndf_frame: np.ndarray, thresholds, beta_probs) -> tuple[list[float], list[float]]:
        """
        Get all possible pitch candidates
        """
        base_trough_indices = scipy.signal.argrelmin(cmndf_frame, order=1)[0]
        # troughs = [parabolic_interpolation(cmndf_frame, x) for x in base_trough_indices]
        # troughs = [Trough(*parabolic_interpolation_numba(cmndf_frame, x)) for x in base_trough_indices]

        troughs = [PYin.parabolic_interpolation_numba(cmndf_frame, i) for i in base_trough_indices]

        trough_x_vals = np.array([trough[0] for trough in troughs])
        trough_y_vals = np.array([trough[1] for trough in troughs])

        trough_thresholds = np.less.outer(trough_y_vals, thresholds)
        trough_ranks = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)

        BOLTZMANN_PARAM = 2.0
        trough_prior = scipy.stats.boltzmann.pmf(trough_ranks, BOLTZMANN_PARAM, n_troughs)
        trough_prior[~trough_thresholds] = 0

        probs = PYin.compute_trough_probabilities(trough_prior, trough_thresholds, beta_probs, no_trough_prob=0.01)

        SAMPLE_RATE = 44100

        trough_frequencies = []
        trough_probabilities = []
        for i, trough in enumerate(trough_y_vals):
            trough_freq = SAMPLE_RATE / trough_x_vals[i]
            trough_frequencies.append(trough_freq)
            trough_prob = probs[i]
            trough_probabilities.append(trough_prob)

        return trough_frequencies, trough_probabilities

    def calculate_alpha_beta(mean_threshold, total=20):
        """
        Calculate alpha and beta for a beta distribution given a desired mean and a total sum of alpha and beta.
        """
        alpha = mean_threshold * total
        beta = total - alpha
        return alpha, beta

    def pyin(audio_data: np.ndarray, mean_threshold: float = 0.3):
        """
        The Probabilistic YIN algorithm for pitch estimation.
        """

        print("Starting pYIN algorithm...")

        sr = 44100
        frame_size = 2048*2
        hop_size = 128

        audio_data = PYin.high_pass_irr_filter(audio_data, sr)

        N_THRESHOLDS = 100
        thresholds = np.linspace(0, 1, N_THRESHOLDS)
        beta_thresholds = np.linspace(0, 1, N_THRESHOLDS + 1)

        # Create beta distribution centered around the desired mean threshold
        alpha, beta = PYin.calculate_alpha_beta(mean_threshold)
        beta_cdf = scipy.stats.beta.cdf(beta_thresholds, alpha, beta) # Why 2, 18?
        beta_probs = np.diff(beta_cdf)
        
        pitches = []
        most_likely_pitches = []
        num_frames = (len(audio_data) - frame_size) // hop_size
        for frame_idx in range(num_frames):
            # Print the frame count in place
            print(f"\rProcessing frame {frame_idx + 1}/{num_frames}", end='')

            i = frame_idx * hop_size
            time = i / sr
            # Compute the CMNDF function for each frame
            audio_frame = audio_data[i:i+frame_size]
            cmndf_frame, _ = PYin.cmndf(audio_frame, frame_size // 2)
            
            # Compute all possible pitch candidates for the frame
            trough_freqs, trough_probs = PYin.pitch_probabilities(cmndf_frame, thresholds, beta_probs)

            max_prob = 0
            most_likely_pitch = None

            for trough_freq, trough_prob in zip(trough_freqs, trough_probs):
                pitch = Pitch(time=time, frequency=trough_freq, probability=trough_prob)
                pitches.append(pitch)
                if trough_prob > max_prob:
                    max_prob = trough_prob
                    most_likely_pitch = pitch

            most_likely_pitches.append(most_likely_pitch)
            
        print("\nDone!")
        return pitches, most_likely_pitches


# def snac_fct(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
#     """
#     SNAC function implementation from McLeod's thesis.
#     """
#     x = np.array(audio_frame, np.float64)  # Ensure float64 precision
#     w = x.size
#     tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

#     # Compute the autocorrelation and power spectrum
#     sdf, power_spec = square_difference_fct(x, tau_max)

#     # Compute m'(tau)
#     m_0 = 2*np.sum(x ** 2) # initial m'(0)

#     # Compute m'(tau) for each possible tau (McLeod 3.3.4)
#     m_primes = np.zeros(tau_max)
#     m_primes[0] = m_0

#     for tau in range(1, tau_max):
#         m_primes[tau] = m_primes[tau-1] - x[tau-1]**2 + x[w-tau]**2

#     # Slice m_primes to only contain valid overlapping values
#     m_primes = m_primes[:w-tau_max]

#     # Compute the SNAC function
#     snac = 1 - sdf / m_primes

#     return snac, power_spec