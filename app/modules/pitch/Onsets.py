import essentia.standard as es
import essentia
import numpy as np
import pandas as pd
from math import ceil

from app.modules.audio.AudioData import AudioData
from app.modules.pitch.Pitch import Pitch

class Onsets:
    def __init__(self):
        self.w = es.Windowing(type='hann')
        self.fft = es.FFT()
        self.c2p = es.CartesianToPolar()
        self.od_complex = es.OnsetDetection(method='complex')
            
    def detect_onsets(self, audio_data: AudioData) -> np.ndarray[float]:
        """
        Detects onsets in the audio data using Essentia's complex onset 
        detection algorithm.

        Args:
            audio_data (AudioData): User's audio data to detect onsets from

        Returns:    
            np.ndarray: Array of onset times in seconds
        """
        print("Detecting onsets...")
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
        print("Onset detection complete.")
        return onset_times
    
    
    def prev_note_pitch(self, pitch_list: list[Pitch], current_idx: int, window_size: int = 15):
        """Find the previous note pitch in a list of pitches, based on a median of up to the last 15 midi_num values."""

        # print(f"Finding previous note for index: {current_idx}")
        
        # Get the sublist of pitches before the current index
        if current_idx < window_size:
            # print(f"Current index {current_idx} is less than window size {window_size}, taking all previous pitches")
            sublist = pitch_list[:current_idx]
        else:
            # print(f"Taking last {window_size} pitches before current index {current_idx}")
            sublist = pitch_list[current_idx - window_size:current_idx]
        
        # Get the freq values from the sublist
        freqs = [pitch.frequency for pitch in sublist]
        
        # print(f"Sublist frequencies: {freqs}")
        
        # Compute and return the median of the available pitches
        if freqs:
            median_freq = np.median(freqs)
            # print(f"Median frequency: {median_freq}")
            return median_freq
        else:
            # If no previous pitches exist, return the current pitch
            # print(f"No previous pitches, returning current pitch frequency: {pitch_list[current_idx].frequency}")
            return pitch_list[current_idx].frequency
        
    # def find_filter_bounds(onset)
        
    def next_note_pitch(self, pitch_list: list[Pitch], current_idx: int, window_size: int = 15):
        """Find the next note pitch in a list of pitches, based on a median of up to the next 15 midi_num values."""
        
        # print(f"Finding next note for index: {current_idx}")
        
        # Get the sublist of pitches after the current index
        if current_idx + window_size >= len(pitch_list):
            # print(f"Current index {current_idx} is less than window size {window_size} from the end, taking all next pitches")
            sublist = pitch_list[current_idx:]
        else:
            # print(f"Taking next {window_size} pitches after current index {current_idx}")
            sublist = pitch_list[current_idx:current_idx + window_size]
        
        # Get the freq values from the sublist
        freqs = [pitch.frequency for pitch in sublist]
        
        # print(f"Sublist frequencies: {freqs}")
        
        # Compute and return the median of the available pitches
        if freqs:
            median_freq = np.median(freqs)
            # print(f"Median frequency: {median_freq}")
            return median_freq
        else:
            # If no next pitches exist, return the current pitch
            # print(f"No next pitches, returning current pitch frequency: {pitch_list[current_idx].frequency}")
            return pitch_list[current_idx].frequency


    def onset_annotation(self, onsets: np.ndarray[float], pitch_list: list[Pitch], window_size: int = 15):
        """Annotate onsets with the previous note pitch. Returns a smaller note_df"""
        
        # Ensure the input data is valid
        print(f"Annotating {len(onsets)} onsets with {len(pitch_list)} pitches")
        
        notes = []
        annotated_onsets = []  # Renamed to avoid shadowing the input 'onsets'
        frequencies = []
        onset_indices = []
        
        for idx, onset in enumerate(onsets):
            # Find corresponding index into pitch_list
            onset_idx = min(range(len(pitch_list)), key=lambda i: abs(pitch_list[i].time - onset))
            # print(f"Onset: {onset}, Closest pitch index: {onset_idx}, Pitch time: {pitch_list[onset_idx].time}")
            
            # Find the previous note pitch based on a rolling median
            prev_pitch = self.prev_note_pitch(pitch_list, onset_idx, window_size)
            
            # Append the annotated onset to the list
            midi_num = pitch_list[0].freq_to_midi(prev_pitch)
            # print(f"Previous pitch frequency: {prev_pitch}, Converted to MIDI: {midi_num}")
            notes.append(midi_num)

            window_size_sec = window_size / 44100
            annotated_onsets.append(onset - window_size_sec)  # Time of the median
            frequencies.append(prev_pitch)
            onset_indices.append(idx)

        # Create a DataFrame to store the annotated onsets
        mini_notedf = {
            'time': annotated_onsets,
            'midi_num': notes,
            'frequency': frequencies,
            'onset_idx': onset_indices
        }
        mini_notedf = pd.DataFrame(mini_notedf)
        
        return mini_notedf

        
    @staticmethod
    def pitch_onsets(pitch_list: list[Pitch], window_size: int=15, threshold: float=0.75):
        """Detect different-enough new pitches based on a rolling median of pitch_bin."""

        print(f"Segmenting notes with window_size={window_size} and threshold={threshold}...")

        # Rolling median on pitch_bin
        rolling_medians = pd.Series([pitch.midi_num for pitch in pitch_list]).rolling(window=window_size).median()

        # Prepare lists for note events
        notes = []
        onsets = []
        frequencies = []

        MAIN_LOWER_BOUND = window_size - 1
        MAIN_UPPER_BOUND = len(rolling_medians) - window_size - 1
        HOP_SIZE = window_size

        last_onset = 0

        for i in range(MAIN_LOWER_BOUND, MAIN_UPPER_BOUND, HOP_SIZE):
            current_median = rolling_medians.iloc[i]
            next_median = rolling_medians.iloc[i + HOP_SIZE]

            if abs(next_median - current_median) >= threshold:
                # Subroutine to find the two points with the greatest difference
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
        return note_df

    
    @staticmethod
    def rolling_median_notes2(user_pitchdf: pd.DataFrame, window_size: int=15, threshold: float=0.75):
        """Detect different-enough new pitches based on a rolling median."""

        print(f"Segmenting notes with window_size={window_size} and threshold={threshold}...")
        rolling_medians = user_pitchdf['midi_pitch'].rolling(window=window_size).median()

        # Detect new notes
        notes = []
        onsets = []
        frequencies = []

        # NAIVE: Start with first median pitch as the first 'note'
        #TODO: Make so there needs to be a certain number of similar enough pitches even for this first note

        MAIN_LOWER_BOUND = window_size - 1
        MAIN_UPPER_BOUND = len(rolling_medians) - window_size - 1
        HOP_SIZE = window_size

        last_onset = 0

        for i in range(MAIN_LOWER_BOUND, MAIN_UPPER_BOUND, HOP_SIZE):
            current_median = rolling_medians.iloc[i]
            next_median = rolling_medians.iloc[i + HOP_SIZE]

            if abs(next_median - current_median) >= threshold:
                # Subroutine to find the two points with greatest difference
                # Note rolling() considers window bounds as the rightmost point (inclusive) - window_size
                WINDOW_LOWER_BOUND = i - ceil(window_size/2)
                WINDOW_UPPER_BOUND = i + ceil(window_size/2)
                largest_diff = (-1, 0) # (index, difference)
                for j in range(WINDOW_LOWER_BOUND, WINDOW_UPPER_BOUND):
                    pitch_diff = abs(user_pitchdf['midi_pitch'].iloc[j] - user_pitchdf['midi_pitch'].iloc[j+1])
                    if pitch_diff > largest_diff[1]:
                        largest_diff = (j, pitch_diff)
                
                notes.append(current_median)
                onsets.append(last_onset) # starts with 0
                frequencies.append(user_pitchdf['frequency'].iloc[i])

                # Update last onset
                last_onset = user_pitchdf['time'].iloc[largest_diff[0]]

        # create a note_df
        note_data = {
            'time': onsets,
            'midi_pitch': notes,
            'frequency': frequencies
        }
        note_df = pd.DataFrame(note_data)
        return note_df