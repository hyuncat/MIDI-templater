import numpy as np
import time
import librosa
import scipy.signal as signal

# want to compare two different pitch detection systems to each other (singscope)
# want to play back the direct output of our detected pitches

def diff_func(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast difference function implementation using numpy's FFT.
    Step 1 of Yin algorithm, corresponding to equation (6) in Cheveigne, Kawahara 2002.
    Adopted from https://github.com/patriceguyot/Yin

    Args:
        audio_frame: The current frame of audio samples in Yin algorithm
        tau_max: The time-lag to check for in the difference function

    Returns: 
        frame_diffs: An array of size tau_max with the difference function values
    """

    audio_frame = np.array(audio_frame, np.float64) # Ensure float64 precision
    window_size = audio_frame.size
    tau_max = min(tau_max, window_size) # Ensure tau_max is within the window size

    # Cumulative sum of squares of each sample in the signal
    audio_cumsum = (audio_frame**2).cumsum()
    audio_cumsum = np.concatenate((np.array([0.]), audio_cumsum)) # Prepend 0 for easier indexing
    
    # Pad the audio signal array by the minimum power of 2 which
    # is larger than the window size + tau_max and makes the fft a 'nice' size
    # (for efficient fft computation)
    min_fft_size = window_size + tau_max # (pad by >tau_max for frame end)

    p2 = (min_fft_size // 32).bit_length() 
    nice_fft_sizes = (16, 18, 20, 24, 25, 27, 30, 32) 
    size_pad = min(size*(2**p2) for size in nice_fft_sizes if size * 2**p2 >= min_fft_size)

    # Decompose the signal into its frequency components
    fft_frame = np.fft.rfft(audio_frame, size_pad) # Use only real part of the FFT (faster)
    
    # Convolution of the signal with a reversed version of itself
    # (autocorrelation)
    convolution = np.fft.irfft(fft_frame * fft_frame.conjugate())[:tau_max]

    # Get the cumulative sum of squares for progressively shorter sections
    # of the signal as tau increases.
    # window_size - tau_max --> window_size (in reverse)
    #TODO: more descriptive name
    reverse_cumsum = audio_cumsum[window_size:window_size - tau_max:-1]

    # Get the total cumulative sum of squares for the entire window (from start to window_size).
    total_cumsum = audio_cumsum[window_size]

    # Slice the cumulative sum array to get the sum of squares from the start of the signal to each tau_max.
    # This gives the cumulative sum of squares for each possible tau (from 0 to tau_max).
    tau_cumsums = audio_cumsum[:tau_max]

    # Get how different the signal to itself is for each lag in frame
    frame_diffs = reverse_cumsum + total_cumsum - tau_cumsums - 2*convolution

    # Calculate the amplitude for each tau value (period)
    amplitudes = np.sqrt(reverse_cumsum / np.arange(1, tau_max + 1))

    # Convert the amplitudes to decibels
    amplitudes_db = librosa.amplitude_to_db(amplitudes, ref=np.max(amplitudes))

    return frame_diffs, amplitudes_db


def cumulative_mean_normalize(frame_diffs: np.ndarray, tau_max: int) -> np.ndarray:
    """
    Cumulative Mean Normalized Difference Function (CMNDF) computation.
    Step 2 of Yin algorithm, corresponding to equation (8) in Cheveigne, Kawahara 2002.
    Adopted from https://github.com/patriceguyot/Yin

    Args:
        frame_diffs: The array of difference function values for each tau
        tau_max: The time-lag to check for in the difference function

    Returns:
        cmndf: A normalized array of size tau_max with the CMNDF values.
    """    

    cmndf = frame_diffs[1:] * range(1, tau_max) / np.cumsum(frame_diffs[1:]).astype(float)
    cmndf = np.concatenate((np.array([1.]), cmndf))

    return cmndf


def parabolic_interpolation(y_vals: np.ndarray, x: int) -> tuple[float, float]:
    """
    Perform parabolic interpolation around a minimum point.
    
    Args:
        y_vals: A 1D array of y-values (e.g., CMNDF values).
        x: The index of the minimum point in y_vals.

    Returns:
        The x & y coordinate of the interpolated minimum, which can be a float.
    """
    if x <= 0 or x >= len(y_vals) - 1:
        return float(x)  # No interpolation possible at the boundaries
    
    alpha = y_vals[x - 1]
    beta = y_vals[x]
    gamma = y_vals[x + 1]

    # Interpolated x-coordinate of the parabola's vertex
    denominator = 2 * (alpha - 2 * beta + gamma)
    if denominator == 0:
        return float(x), beta
    
    x_interpolated = x + (alpha - gamma) / denominator
    y_interpolated = beta - (alpha - gamma) * (alpha - gamma) / (4 * denominator)
    
    return x_interpolated, y_interpolated

def get_pitch_period(cmndf: np.ndarray, min_period: int, max_period: int, max_diff: float) -> tuple[float, float]:
    """
    Get the fundamental period of a frame based on the CMNDF array.
    The "absolute threshold" step in the Yin paper, with parabolic interpolation.
    
    Args:
        cmndf: The Cumulative Mean Normalized Difference Function array.
        min_period: The minimum period to consider.
        max_period: The maximum period to consider.
        max_diff: The maximum value of CMNDF to consider a candidate.

    Returns:
        The interpolated period (tau) corresponding to the pitch.
    """
    for tau in range(min_period, max_period):
        # If the value of the CMNDF function is below 'max_diff' threshold
        if cmndf[tau] < max_diff:
            while tau + 1 < max_period:
                # Find a local minimum period and call it the pitch
                if cmndf[tau + 1] > cmndf[tau]:
                    # Perform parabolic interpolation around the local minimum to get the pitch period
                    return parabolic_interpolation(cmndf, tau)
                tau += 1
    return 0.0, 0.0

def get_pitch_periods(cmndf: np.ndarray, min_period: int, max_period: int, max_diff: float, num_candidates: int = 1) -> list[int]:
    """
    Get the top possible pitch candidates based on the CMNDF array.
    
    Args:
        cmndf: The Cumulative Mean Normalized Difference Function array.
        min_period: The minimum period to consider.
        max_period: The maximum period to consider.
        max_diff: The maximum value of CMNDF to consider a candidate.
        num_candidates: The number of top pitch candidates to return.

    Returns:
        A list of possible pitch periods (tau) within the period range.
    """
    candidates = []

    for tau in range(min_period, max_period):
        # Check if current tau is a local minimum and below the max_diff threshold
        if cmndf[tau] < max_diff:
            if tau + 1 < max_period and cmndf[tau + 1] > cmndf[tau]:
                candidates.append((tau, cmndf[tau]))
    
    # Sort candidates by the CMNDF value (ascending) and extract the top ones
    candidates = sorted(candidates, key=lambda x: x[1])[:num_candidates]
    
    # Return only the periods (tau) of the top candidates
    return [candidate[0] for candidate in candidates]

def pitch_yin(audio_data: np.ndarray, sample_rate: int=44100, 
              frame_size: int=2048, hop_size: int=128,
              min_freq: int=198, max_freq: int=5000, max_diff: float=.5) -> tuple:
    """
    Yin pitch detection algorithm implementation.
    """
    
    print("Computing pitches...")
    start_time = time.time()

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

    tau_min = int(sample_rate / max_freq)
    tau_max = int(sample_rate / min_freq)

    time_scale = range(0, len(audio_data)-frame_size, hop_size)
    times = [t/(float(sample_rate)) for t in time_scale]
    frames = [audio_data[t:t+frame_size] for t in time_scale]

    pitches = np.zeros(len(frames))
    harmonic_rates = np.zeros(len(frames))
    argmins = np.zeros(len(frames)) 
    amplitudes = np.zeros(len(frames))
    
    for i, frame in enumerate(frames):
        frame_diffs, amps = diff_func(frame, tau_max)
        cmndf = cumulative_mean_normalize(frame_diffs, tau_max)
        pitch_period, harmonic_rate = get_pitch_period(cmndf, tau_min, tau_max, max_diff)

        # curr_pitches = []
        # curr_hrs = []
        # curr_amps = []
        # for pitch_period in pitch_periods:
        #     if np.argmin(cmndf) > tau_min:
        #         argmins[i] = float(sample_rate) / np.argmin(cmndf)
        #     if pitch_period != 0:
        #         curr_pitches.append(float(sample_rate) / pitch_period)
        #         curr_hrs.append(cmndf[pitch_period])
        #         curr_amps.append(amps[pitch_period])
        #     else:
        #         curr_pitches.append(0)
        #         curr_hrs.append(cmndf[np.argmin(cmndf)])
        #         curr_amps.append(amps[np.argmin(cmndf)])
            
            # append the list
            # pitches[i] = curr_pitches
            # harmonic_rates[i] = curr_hrs
            # amplitudes[i] = curr_amps
        
        if np.argmin(cmndf) > tau_min:
            argmins[i] = float(sample_rate) / np.argmin(cmndf)
        if pitch_period != 0:
            pitches[i] = float(sample_rate) / pitch_period
            harmonic_rates[i] = harmonic_rate
            amplitudes[i] = amps[int(pitch_period)]
        else:
            pitches[i] = 0

            min_cmndf_index = np.argmin(cmndf)
            harmonic_rates[i] = cmndf[min_cmndf_index]
            amplitudes[i] = amps[min_cmndf_index]
        
    end_time = time.time()
    print(f"Done! Took {end_time - start_time:.2f} seconds.")
    return pitches, harmonic_rates, amplitudes, argmins, times


