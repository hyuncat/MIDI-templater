import essentia.standard as es
import numpy as np

class PitchAnalyzer:
    def __init__(self, samplerate=44100):
        self.samplerate = samplerate
        self.pitchYin = es.PitchYin(frameSize=2048,
                                    interpolate=True,
                                    maxFrequency=5000,
                                    minFrequency=150,
                                    sampleRate=samplerate,
                                    tolerance=0.15)

    def analyze_pitch(self, audio_data):
        audio_data_float = audio_data.astype(np.float32) / 32768.0
        pitch, pitchConfidence = self.pitchYin(audio_data_float)
        return pitch, pitchConfidence
