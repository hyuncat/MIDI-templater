import sounddevice as sd
import numpy as np
import time
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex, QWaitCondition
import threading
from .PitchAnalysis import PitchAnalyzer

class SharedAudioData:
    def __init__(self, midi_length_seconds, samplerate=44100, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.buffer = np.zeros(int(samplerate * midi_length_seconds * channels), dtype=np.float32)
        self.lock = threading.Lock()  # To ensure thread-safe access to the buffer

    def write_data(self, start_time, end_time, data):
        start_idx = int(start_time * self.samplerate * self.channels)
        end_idx = int(end_time * self.samplerate * self.channels)
        concatenated_recording = data.flatten()[:end_idx - start_idx]

        # Ensure the length of the concatenated recording matches the buffer slice
        if concatenated_recording.shape[0] > end_idx - start_idx:
            concatenated_recording = concatenated_recording[:end_idx - start_idx]
        elif concatenated_recording.shape[0] < end_idx - start_idx:
            concatenated_recording = np.pad(concatenated_recording, (0, end_idx - start_idx - concatenated_recording.shape[0]), 'constant')

        with self.lock:
            self.buffer[start_idx:end_idx] = concatenated_recording

    def read_data(self, start_time, end_time):
        start_idx = int(start_time * self.samplerate * self.channels)
        end_idx = int(end_time * self.samplerate * self.channels)
        with self.lock:
            return self.buffer[start_idx:end_idx]
        
class AudioRecorderThread(QThread):
    recording_started = pyqtSignal(float)
    recording_stopped = pyqtSignal(float)
    pitch_data_updated = pyqtSignal(float, float)

    def __init__(self, shared_data):
        super().__init__()
        self.shared_data = shared_data
        self.is_recording = False
        self.start_time = 0
        self.current_recording = []
        self.pitch_analyzer = PitchAnalyzer()
        self.pitch_data = {}

    def record_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.current_recording.append(indata.copy())
            self.add_pitch_data(indata)

    def add_pitch_data(self, audio_data):
        audio_data_float = audio_data.astype(np.float32).flatten()  # Ensure it is a 1D array
        pitch, pitchConfidence = self.pitch_analyzer.analyze_pitch(audio_data_float)
        if pitchConfidence > 0.5:
            current_time = self.get_current_time()
            self.pitch_data[current_time] = pitch
            print(f"Time: {current_time} Pitch: {pitch} Midi: {(12*np.log2(pitch/440)+69)} Hz, Confidence: {pitchConfidence}")
            self.pitch_data_updated.emit(current_time, pitch)

    def get_current_time(self):
        return self.start_time + len(self.current_recording) * len(self.current_recording[0]) / self.shared_data.samplerate


    def run(self):
        self.stream = sd.InputStream(samplerate=self.shared_data.samplerate, channels=self.shared_data.channels, callback=self.record_callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def start_recording(self, start_time):
        self.start_time = start_time
        self.current_recording = []
        self.is_recording = True
        self.recording_started.emit(self.start_time)
        print("Recording started...")

    def stop_recording(self, end_time):
        self.is_recording = False
        if self.start_time is not None:
            concatenated_recording = np.concatenate(self.current_recording, axis=0).flatten()
            self.shared_data.write_data(self.start_time, end_time, concatenated_recording)
            self.recording_stopped.emit(end_time)
            print(f"Recording stopped. Duration: {end_time - self.start_time:.2f} seconds")
            self.start_time = None
            self.current_recording = []
    
    def get_is_recording(self):
        return self.is_recording

class AudioPlaybackThread(QThread):
    def __init__(self, shared_data):
        super().__init__()
        self.shared_data = shared_data
        self.is_paused = False
        self.playback_position = 0
        self.playback_time = 0

    def run(self):
        while self.playback_position < len(self.shared_data.buffer):
            if not self.is_paused:
                playback_data = self.shared_data.buffer[self.playback_position:]
                sd.play(playback_data, samplerate=self.shared_data.samplerate)
                sd.wait()
                break
            self.msleep(100)

    def play_audio(self, start_from):
        self.playback_position = int(start_from * self.shared_data.samplerate)
        self.is_paused = False
        self.start()

    def pause_audio(self):
        self.is_paused = True
        sd.stop()

    def get_is_playing(self):
        return not self.is_paused
