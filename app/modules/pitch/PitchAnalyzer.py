import essentia.standard as es
import numpy as np
from app.config import AppConfig

class PitchAnalyzer:
    def __init__(self):
        """
        Class for pitch analysis using Essentia's PitchYin algorithm.
        The frequency range is tailored for violin pitch detection.
        """
        MIN_VIOLIN_FREQ = 196
        MAX_VIOLIN_FREQ = 5000
        self.pitchYin = es.PitchYin(
            frameSize=AppConfig.FRAME_SIZE,
            interpolate=True,
            maxFrequency=MAX_VIOLIN_FREQ,
            minFrequency=MIN_VIOLIN_FREQ,
            sampleRate=AppConfig.SAMPLE_RATE,
            tolerance=0.15 # Tolerance for peak detection (default)
        )

        FRAME_SIZE = 2048
        HOP_SIZE = 128
        self.pitch_melodia = es.PitchMelodia(
            frameSize=FRAME_SIZE, 
            hopSize=HOP_SIZE,
            # guessUnvoiced=True,
            minFrequency=MIN_VIOLIN_FREQ,
            maxFrequency=MAX_VIOLIN_FREQ,
            sampleRate=AppConfig.SAMPLE_RATE
        )
# pitch_values, pitch_confidences = pitch_melodia(user_audio)


    def get_frame_pitch(self, audio_frame: np.ndarray) -> tuple[float, float]:
        """
        Normalizes a given audio frame from and estimates its pitch (with some confidence). 
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
        normalized_audio_frame = audio_frame.astype(np.float32) / FLOAT32_RANGE
        pitch, confidence = self.pitchYin(normalized_audio_frame)
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
            pitch, confidence = self.pitchYin(frame)
            pitch_values.append(pitch)
            pitch_confidences.append(confidence)
            
            # Calculate the time for the current frame
            time = frame_index * AppConfig.HOP_SIZE / AppConfig.SAMPLE_RATE
            pitch_times.append(time)
            
            frame_index += 1

        return np.array(pitch_values), np.array(pitch_confidences), np.array(pitch_times)
    
    def pitch_melodia(self, audio_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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