import scipy.signal as signal
import numpy as np
from app.config import AppConfig

class Filter:
    def __init__(self):
        pass

    @staticmethod
    def high_pass_irr_filter(audio_data: np.ndarray, cutoff_freq=196, 
                             sample_rate: int=AppConfig.SAMPLE_RATE) -> np.ndarray:
        """
        High pass IRR filter to lower intensity of low frequency noise
        Based on method from McLeod's thesis.
        """
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