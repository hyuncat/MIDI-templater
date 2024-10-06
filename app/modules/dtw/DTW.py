
import numpy as np
import pandas as pd
import librosa
import scipy
from dtw import * # fix this later lol
import pretty_midi
from typing import Optional, List
from dataclasses import dataclass, field
import warnings

from app.modules.audio.AudioData import AudioData
from app.modules.midi.MidiData import MidiData
from app.config import AppConfig
from app.modules.pitch.Pitch import Pitch

class CQT:
    MIN_VIOLIN_FREQ = 196
    N_BINS = 60 # 5 octaves, 12 bins / octave
    HOP_LENGTH = 1024
    FRAME_SIZE = AppConfig.FRAME_SIZE
    TUNING = 0.0

    def __init__(self):
        pass

    @staticmethod
    def extract_cqt_frame(audio_frame: np.ndarray, sample_rate=None) -> pd.DataFrame:
        """
        Compute CQT of an audio signal on a single frame, perform log-amplitude scaling,
        normalize, and return the CQT vector for the frame.

        Args:
            audio_data (np.ndarray): Audio data to compute CQT of
            sample_rate (int, optional): Sample rate of the audio data. Defaults to AppConfig.SAMPLE_RATE.

        Returns:
            pd.DataFrame: A DataFrame containing the time of each frame and the averaged cqt_norm for each frame.
        """

        if sample_rate is None:
            sample_rate = AppConfig.SAMPLE_RATE

        # Suppress UserWarning and ComplexWarning
        warnings.filterwarnings('ignore')

        # Compute CQT
        cqt = librosa.cqt(
            audio_frame, 
            sr=sample_rate,
            fmin=CQT.MIN_VIOLIN_FREQ, 
            n_bins=CQT.N_BINS, 
            hop_length=len(audio_frame)+2, # used to be dtw.hop_size
            tuning=CQT.TUNING
        )

        # Compute the time of each frame
        # times = librosa.frames_to_time(
        #     np.arange(cqt.shape[1]), 
        #     sr=sample_rate, 
        #     hop_length=DTW.HOP_LENGTH # corresponds to frame size
        # )

        # Convert CQT to float32 to save space
        cqt = cqt.astype(np.float32)

        # Compute log-amplitude and normalize the CQT
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt_norm = librosa.util.normalize(cqt_db, norm=2)

        return cqt_norm
    
class DTW:
    def __init__(self):
        pass

    def cqt_onset_df(audio_data: np.ndarray, onset_df: pd.DataFrame):
        """
        Accepts an onset_df with timestamps for when onsets occur and annotates
        each one with the CQT vector of audio at that given frame
        """

        onset_df['cqt_norm'] = None
        for i, onset in onset_df.iterrows():
            onset_time = onset['time']
            onset_idx = int(onset_time * AppConfig.SAMPLE_RATE)

            FRAME_SIZE = int(AppConfig.FRAME_SIZE)
            audio_frame = audio_data[onset_idx:onset_idx+FRAME_SIZE]

            cqt_norm = CQT.extract_cqt_frame(audio_frame)
            onset_df.loc[i, 'cqt_norm'] = cqt_norm
        
        return onset_df
