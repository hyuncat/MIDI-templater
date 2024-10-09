
import numpy as np
import pandas as pd
import librosa
import scipy
from dtw import * # fix this later lol
import pretty_midi
from typing import Optional, List
from dataclasses import dataclass, field
import warnings

from app.modules.core.audio.AudioData import AudioData
from app.modules.core.midi.MidiData import MidiData
from app.config import AppConfig
from app.modules.processing.pda.Pitch import Pitch
from app.modules.processing.dtw.OnsetDf import UserOnsetDf, MidiOnsetDf
    
class DTW:
    def __init__(self):
        pass

    @staticmethod
    def align(user_onset_df: UserOnsetDf, midi_onset_df: MidiOnsetDf):
        # Create distance matrix
        user_cqt = np.stack(user_onset_df.onset_df['cqt_norm'].values)
        midi_cqt = np.stack(midi_onset_df.onset_df['cqt_norm'].values)

        distance_matrix = scipy.spatial.distance.cdist(midi_cqt, user_cqt, metric='cosine')

        # Compute the alignment
        window_args = {'window_size': 100}
        alignment = dtw(
            distance_matrix,
            keep_internals=True,
            step_pattern=symmetric1,
            window_type='sakoechiba',
            window_args=window_args
        )

        # Print some statistics (from dtw-python documentation)
        # Compute the mean alignment error
        mean_error = np.mean(np.abs(alignment.index1 - alignment.index2))

        # Print some information about the alignment
        print("DTW alignment computed.")
        print(f"Distance: {alignment.distance}") # unit = cosine distance
        print(f"Mean alignment error: {mean_error}") # unit = frames

        return alignment
    
    @staticmethod
    def align_df(alignment, user_onset_df: UserOnsetDf, midi_onset_df: MidiOnsetDf):
        """Create a dataframe parsing the alignment result into more meaningful results."""
        aligned_user = user_onset_df.onset_df.iloc[alignment.index2].reset_index(drop=True)
        aligned_midi = midi_onset_df.onset_df.iloc[alignment.index1].reset_index(drop=True)

        flat_align_df = pd.DataFrame({
            'midi_time': aligned_midi['time'],
            'user_time': aligned_user['time']
        })

        grouped = flat_align_df.groupby('midi_time')['user_time'].apply(list).reset_index()
        align_df = pd.DataFrame({
            'midi_time': grouped['midi_time'],
            'user_time': grouped['user_time'].apply(lambda x: x if len(x) > 0 else [None])
        })

        return align_df