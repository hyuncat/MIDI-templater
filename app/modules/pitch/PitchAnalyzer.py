import essentia.standard as es
import essentia
import numpy as np
import pandas as pd
from math import ceil

from app.config import AppConfig
from app.modules.audio.AudioData import AudioData

class PitchAnalyzer:
    def __init__(self):
        """
        Class for pitch analysis using Essentia's PitchYin algorithm.
        The frequency range is tailored for violin pitch detection.
        """
        MIN_VIOLIN_FREQ = 196
        MAX_VIOLIN_FREQ = 5000
        self.pitch_yin = es.PitchYin(
            frameSize=AppConfig.FRAME_SIZE,
            interpolate=True,
            maxFrequency=MAX_VIOLIN_FREQ,
            minFrequency=MIN_VIOLIN_FREQ,
            sampleRate=AppConfig.SAMPLE_RATE,
            tolerance=0.15 # Tolerance for peak detection (default)
        )

        self.pitch_melodia = es.PitchMelodia(
            frameSize=AppConfig.FRAME_SIZE,
            hopSize=AppConfig.HOP_SIZE,
            # guessUnvoiced=True,
            minFrequency=MIN_VIOLIN_FREQ,
            maxFrequency=MAX_VIOLIN_FREQ,
            sampleRate=AppConfig.SAMPLE_RATE
        )

        # Onset detection algorithms
        self.od_complex = es.OnsetDetection(method='complex', sampleRate=AppConfig.SAMPLE_RATE)
        # Auxiliary algorithms to compute magnitude and phase
        self.w = es.Windowing(type='hann', size=AppConfig.FRAME_SIZE)
        self.fft = es.FFT(size=AppConfig.FRAME_SIZE)  # Outputs a complex FFT vector.
        self.c2p = es.CartesianToPolar()  # Converts it into a pair of magnitude and phase vectors.


    def get_pitch(self, audio_frame: np.ndarray) -> tuple[float, float]:
        """
        Returns a single pitch + confidence from a frame of audio data.
        This function is called every time AudioRecorder._callback() is triggered.
        @param:
            - audio_frame (np.ndarray): frame of audio data
                -> corresponds to 'indata' from AudioRecorder._callback()
        @return:
            - pitch (float): estimated pitch in Hz
            - confidence (float): confidence in pitch estimation [0, 1]
        """
        # Normalize each sample in the frame to the range [-1, 1]
        FLOAT32_RANGE = 32768.0
        normalized_audio_frame = audio_frame.astype(np.float32) / FLOAT32_RANGE # this may not be necessary lol
        pitch, confidence = self.pitch_yin(normalized_audio_frame)
        return pitch, confidence

    def get_buffer_pitch(self, audio_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimates the pitches for each frame in an array of frames.
        @param:
            - audio_buffer (np.ndarray): entire audio buffer
                -> corresponds to 'self.audio_buffer' from AudioRecorder
        @return:
            - pitches (np.ndarray): estimated pitches in Hz
            - confidences (np.ndarray): confidence in pitch estimation [0, 1]
            - pitch_times (np.ndarray): times at which each pitch was recorded
        """
        # Apply equal-loudness filter for better pitch results
        audio_buffer = es.EqualLoudness()(audio_buffer)

        pitch_values = []
        pitch_confidences = []
        pitch_times = []

        frame_index = 0
        for frame in es.FrameGenerator(audio_buffer, 
                                       frameSize=AppConfig.FRAME_SIZE, 
                                       hopSize=AppConfig.HOP_SIZE):
            pitch, confidence = self.pitch_yin(frame)
            pitch_values.append(pitch)
            pitch_confidences.append(confidence)
            
            # Calculate the time for the current frame
            time = frame_index * AppConfig.HOP_SIZE / AppConfig.SAMPLE_RATE
            pitch_times.append(time)
            
            frame_index += 1

        return np.array(pitch_values), np.array(pitch_confidences), np.array(pitch_times)
    
    def user_pitchdf(self, audio_data: AudioData):
        """Get all pitches from the audio data"""

        equalized_audio_data = es.EqualLoudness()(audio_data.data)

        frequencies = []
        confidences = []

        print("Starting PitchYin...")
        for frame in es.FrameGenerator(equalized_audio_data, frameSize=AppConfig.FRAME_SIZE, hopSize=128):
            # FLOAT32_RANGE = 32768.0
            # frame = frame.astype(np.float32) / FLOAT32_RANGE
            freq, conf = self.pitch_yin(frame)
            frequencies.append(freq)
            confidences.append(conf)

        print("PitchYin complete.")

        # Pitch is estimated on frames. Compute frame time positions.
        pitch_times = np.linspace(0.0, len(equalized_audio_data)/AppConfig.SAMPLE_RATE, len(frequencies))

        # Equation source: https://www.music.mcgill.ca/~gary/307/week1/node28.html
        midi_pitches = [(12*np.log(freq/220)/np.log(2) + 57) for freq in frequencies]

        pitch_data = {
            'time': pitch_times,
            'frequency': frequencies,
            'midi_pitch': midi_pitches,
            'confidence': confidences
        }

        pitch_df = pd.DataFrame(pitch_data)
        # filter rows where confidence is 0 (recommended by essentia)
        pitch_df = pitch_df[pitch_df['confidence'] > 0] 

        return pitch_df
    
    @staticmethod
    def note_segmentation(user_pitchdf: pd.DataFrame, window_size: int=11, threshold: float=0.75):
        """Detect different-enough new pitches based on a rolling median."""

        print("Segmenting notes...")
        rolling_medians = user_pitchdf['midi_pitch'].rolling(window=window_size).median()

        # Detect new notes
        notes = []
        onsets = []
        frequencies = []

        # NAIVE: Start with first median pitch as the first 'note'
        #TODO: Make so there needs to be a certain number of similar enough pitches even for this first note
        # notes.append(rolling_medians.iloc[window_size - 1])
        # onsets.append(user_pitchdf['time'].iloc[0])
        # frequencies.append(user_pitchdf['frequency'].iloc[0])

        MAIN_LOWER_BOUND = window_size - 1
        MAIN_UPPER_BOUND = len(rolling_medians) - window_size - 1
        HOP_SIZE = window_size

        last_onset = 0

        for i in range(MAIN_LOWER_BOUND, MAIN_UPPER_BOUND, HOP_SIZE):
            current_median = rolling_medians.iloc[i]
            next_median = rolling_medians.iloc[i + HOP_SIZE]

            if abs(next_median - current_median) >= threshold:
                # notes.append(current_median)
                # onsets.append(user_pitchdf['time'].iloc[i])
                # frequencies.append(user_pitchdf['frequency'].iloc[i])

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
    
    @staticmethod
    def group_harmonics(note_df: pd.DataFrame, harmonic_range=0.75):
        MIN_VIOLIN_FREQ = 196
        MAX_VIOLIN_FREQ = 5000

        harmonic_groups = []
        visited = set()
        print("Grouping harmonics...")

        for i, note_row in note_df.iterrows():

            if i in visited: # Ignore notes that have already been grouped
                continue

            frequency = note_row['frequency']

            # Find all harmonics of the note within violin frequency range
            above_harmonics = [frequency * i for i in range(1, 10) if frequency * i <= MAX_VIOLIN_FREQ]
            below_harmonics = [frequency / i for i in range(2, 10) if frequency / i >= MIN_VIOLIN_FREQ]
            harmonics = above_harmonics + below_harmonics # All harmonics to search for

            # Convert harmonics to MIDI pitches
            # https://www.music.mcgill.ca/~gary/307/week1/node28.html
            freq_to_midi = lambda freq: 12*np.log(freq/220)/np.log(2) + 57
            harmonic_midi_pitches = [freq_to_midi(harmonic) for harmonic in harmonics]

            
            harmonic_group = []
            visited.add(i)
            for j, next_note_row in note_df.iterrows():
                if j in visited: # Again ignore notes that have already been grouped
                    continue

                next_midi_pitch = next_note_row['midi_pitch']

                # Check if next_midi_pitch is within a ±0.5 range of any harmonic_midi_pitches
                if any(abs(next_midi_pitch - harmonic_midi_pitch) <= harmonic_range for harmonic_midi_pitch in harmonic_midi_pitches):
                    # Group the notes together
                    harmonic_group.append(next_note_row)
                    visited.add(j)

                else:
                    break # stop searching for consecutive harmonics

            harmonic_groups.append(harmonic_group)

        return harmonic_groups
                    
            

    
    def detect_onsets(self, audio_data: AudioData):
        """Detects onsets in the audio data using Essentia's complex onset detection algorithm."""
        print("Detecting onsets...")
        # Compute both ODF frame by frame. Store results to a Pool.
        pool = essentia.Pool()
        for frame in es.FrameGenerator(audio_data.data, frameSize=1024, hopSize=512):
            magnitude, phase = self.c2p(self.fft(self.w(frame)))
            odf_value = self.od_complex(magnitude, phase)
            pool.add('odf.complex', odf_value)
            # print(f"Frame {i}: Time {frame_time:.4f}s, ODF Value {odf_value}")

        # Detect onset locations
        onsets = es.Onsets()
        onset_times = onsets(essentia.array([pool['odf.complex']]), [1])
        print(f"Onset times: {onset_times}")
        print("Onset detection complete.")
        return onset_times
    
    def detect_onsets2(self, audio_data: AudioData):
        """Detects onsets in the audio data using Essentia's complex onset detection algorithm."""
        print("Detecting onsets...")
        # Compute both ODF frame by frame. Store results to a Pool.
        pool = essentia.Pool()
        for i, frame in enumerate(es.FrameGenerator(audio_data.data, frameSize=AppConfig.FRAME_SIZE, hopSize=AppConfig.HOP_SIZE)):
            magnitude, phase = self.c2p(self.fft(self.w(frame)))
            odf_value = self.od_complex(magnitude, phase)
            pool.add('odf.complex', odf_value)
            # print(f"Frame {i}: Time {frame_time:.4f}s, ODF Value {odf_value}")


        # Detect onset locations
        onsets = es.Onsets()
        onset_times = onsets(essentia.array([pool['odf.complex']]), [1])


        # Debugging: Print onset times and audio length
        print(f"Onset times: {onset_times}")
        audio_length = len(audio_data.data) / AppConfig.SAMPLE_RATE
        print(f"Audio length: {audio_length} seconds")

        print("Onset detection complete.")
        
        # # Write to file to inspect
        # silence_duration = len(audio_data.data) / AppConfig.SAMPLE_RATE
        # silence = np.zeros(int(silence_duration * AppConfig.SAMPLE_RATE), dtype=np.float32)

        # beeps = es.AudioOnsetsMarker(onsets=onset_times, type='beep')(silence)
        # onset_audio = es.StereoMuxer()(audio_data.data, beeps)

        # # Optional: Save the beeped audio to a file
        # import soundfile as sf

        # sf.write('user_fugue_onsets.wav', onset_audio, AppConfig.SAMPLE_RATE)
        return onset_times


    
    def get_pitch_melodia(self, audio_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates the pitches for each frame in an array of frames using the Melodia algorithm.
        @param:
            - audio_buffer (np.ndarray): entire audio buffer
                -> corresponds to 'self.audio_buffer' from AudioRecorder
        @return:
            - pitches (np.ndarray): estimated pitches in Hz
            - pitch_times (np.ndarray): times at which each pitch was recorded
        """
        # Apply equal-loudness filter for better pitch results
        audio_buffer = es.EqualLoudness()(audio_buffer)

        # Use the Melodia algorithm for pitch detection
        pitch_values, pitch_times = self.pitch_melodia(audio_buffer)
        return pitch_values, pitch_times
    

    