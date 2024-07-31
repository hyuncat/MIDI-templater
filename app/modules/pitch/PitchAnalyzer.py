import essentia.standard as es
import numpy as np
import pandas as pd

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
    def note_segmentation(user_pitchdf: pd.DataFrame, window_size=11, threshold=0.75):
        """Detect different-enough new pitches based on a rolling median."""

        rolling_medians = pd.Series(user_pitchdf['midi_pitch']).rolling(window=window_size).median()

        # Detect new notes
        notes = []
        note_times = []

        # Start with first median pitch as the first 'note'
        notes.append(rolling_medians.iloc[0])
        note_times.append(user_pitchdf['time'].iloc[0])

        for i in range(len(rolling_medians) - 1):

            current_median = rolling_medians.iloc[i]
            next_median = rolling_medians.iloc[i + 1]

            if abs(next_median - current_median) >= threshold:
                notes.append(current_median)
                note_times.append(user_pitchdf['time'].iloc[i + 1])

        return notes, note_times

    
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
    

    