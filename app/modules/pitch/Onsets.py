import essentia.standard as es
import essentia
import numpy as np
import pandas as pd
from math import ceil

from app.modules.audio.AudioData import AudioData
from app.modules.pitch.Pitch import Pitch

class OnsetData:
    def __init__(self, audio_data: AudioData, method='complex'):
        self.audio_data = audio_data

        # Essentia onset detection helpers
        self.w = es.Windowing(type='hann')
        self.fft = es.FFT()
        self.c2p = es.CartesianToPolar()
        self.od_complex = es.OnsetDetection(method=method)

        # Onset times from essentia complex OD algorithm
        self.onset_times = self.detect_onsets(self.audio_data)
        
            
    def detect_onsets(self, audio_data: AudioData) -> np.ndarray[float]:
        """
        Detects onsets in the audio data using Essentia's complex onset 
        detection algorithm.

        Args:
            audio_data (AudioData): User's audio data to detect onsets from

        Returns:    
            np.ndarray: Array of onset times in seconds
        """
        print("Detecting onsets...", end='', flush=True)
        # Compute both ODF frame by frame. Store results to a Pool.
        pool = essentia.Pool()
        FRAME_SIZE = 1024
        HOP_SIZE = 512
        for frame in es.FrameGenerator(audio_data.data, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
            magnitude, phase = self.c2p(self.fft(self.w(frame)))
            odf_value = self.od_complex(magnitude, phase)
            pool.add('odf.complex', odf_value)
            # print(f"Frame {i}: Time {frame_time:.4f}s, ODF Value {odf_value}")

        # Detect onset locations
        onsets = es.Onsets()
        onset_times = onsets(essentia.array([pool['odf.complex']]), [1])
        print(" Done!")
        return onset_times

        
    @staticmethod
    def detect_pitch_changes(pitch_list: list[Pitch], window_size: int=15, threshold: float=0.75):
        """
        Detect different-enough new pitches based on a rolling median of pitch_bin.
        We compare the current rolling median to the next one, and when the difference 
        exceeds [threshold], we add it as a new note in final note_df.

        Args:
            pitch_list: A list of Pitch objects
            window_size: Size of the rolling window to compute the median pitch. 
            threshold: Min difference between the current and next rolling median 
                       pitch needed to detect a new pitch change. 

        Returns:
            pd.DataFrame: A DataFrame containing the detected notes with their onset times, MIDI pitch numbers, and frequencies. The DataFrame has the following columns:
                - 'time' (float): The onset time of the detected note in seconds.
                - 'midi_num' (float): The MIDI pitch number of the detected note.
                - 'frequency' (float): The frequency in Hz of the detected note.
        """

        print(f"Segmenting notes with window_size={window_size} and threshold={threshold}...", end='', flush=True)

        # Rolling median on pitch_bin
        rolling_medians = pd.Series([pitch.midi_num for pitch in pitch_list]).rolling(window=window_size).median()

        # Prepare lists for note events
        notes = []
        onsets = []
        frequencies = []

        MAIN_LOWER_BOUND = window_size - 1
        MAIN_UPPER_BOUND = len(rolling_medians) - window_size - 1
        HOP_SIZE = window_size # Compare this window to next window with no overlap

        last_onset = 0

        for i in range(MAIN_LOWER_BOUND, MAIN_UPPER_BOUND, HOP_SIZE):
            current_median = rolling_medians.iloc[i]
            next_median = rolling_medians.iloc[i + HOP_SIZE]

            if abs(next_median - current_median) >= threshold:
                # Subroutine to find the best onset within the window
                # based on the two pitches with the greatest difference
                #TODO: Make subroutine resistant to octave jumps
                WINDOW_LOWER_BOUND = i - ceil(window_size / 2)
                WINDOW_UPPER_BOUND = i + ceil(window_size / 2)
                largest_diff = (-1, 0)  # (index, difference)
                
                for j in range(WINDOW_LOWER_BOUND, WINDOW_UPPER_BOUND):
                    pitch_diff = abs(pitch_list[j].midi_num - pitch_list[j + 1].midi_num)
                    if pitch_diff > largest_diff[1]:
                        largest_diff = (j, pitch_diff)

                # Append the note data
                notes.append(current_median)
                onsets.append(pitch_list[last_onset].time)  # Time of the last onset
                frequencies.append(pitch_list[i].frequency)

                # Update last onset
                last_onset = largest_diff[0]

        # Create a DataFrame to store the detected notes
        note_data = {
            'time': onsets,
            'midi_num': notes,
            'frequency': frequencies
        }
        note_df = pd.DataFrame(note_data)
        print(" Done!")
        return note_df

    @staticmethod
    def combine_onsets(onset_times: np.ndarray, note_df: pd.DataFrame, combine_threshold=0.01) -> pd.DataFrame:
        """
        Args:
            onsets: Array of onset times from librosa
            note_df: Note_df of segmented notes based on pitch changes
            combine_threshold: The max difference (s) between the onset time and the 
                               user note time to be considered the same onset
        """
        # Combine and sort the 'time' columns from both pitch_diff/onset DataFrames
        onset_df = pd.DataFrame({'time': onset_times})
        combined_times = pd.concat(
            [note_df['time'], 
             onset_df['time']]
        ).sort_values().reset_index(drop=True)

        # Remove duplicates within the tolerance
        unique_times = [combined_times[0]]
        for time in combined_times[1:]:
            if not any(np.isclose(time, unique_times, atol=combine_threshold)):
                unique_times.append(time)

        # Create a new DataFrame with combined + unique times
        combined_onset_df = pd.DataFrame({'time': unique_times})

        # Lambda function for finding any df.time values that are close to t
        time_is_close = lambda t, df: any(np.isclose(t, df['time'], atol=combine_threshold))

        # Create boolean columns for whether the time is close to a user note or an onset
        combined_onset_df['pitch_diff'] = combined_onset_df['time'].apply(lambda t: time_is_close(t, note_df))
        combined_onset_df['onset'] = combined_onset_df['time'].apply(lambda t: time_is_close(t, onset_df))

        return combined_onset_df