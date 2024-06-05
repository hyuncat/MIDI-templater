import sys
import numpy as np
import pyaudio
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
import essentia.standard as es
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class AudioRecorder(QWidget):
    def __init__(self):
        self.app = QApplication(sys.argv)
        super().__init__()
        self.setupPlot()
        self.initUI()
        self.initAudio()
        self.pitchYin = self.setupPitchYin()

    def run(self):
        self.show()
        sys.exit(self.app.exec_())

    def initUI(self):
        # Set up the GUI layout
        self.button = QPushButton('Start Recording', self)
        self.button.clicked.connect(self.toggleRecording)
        layout = QVBoxLayout(self)
        layout.addWidget(self.button)

        # Plot setup
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.setWindowTitle('Audio Recorder')
        self.is_recording = False

    def initAudio(self):
        # Set up audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=44100,
                                      input=True,
                                      frames_per_buffer=1024,
                                      stream_callback=self.callback)

    def setupPitchYin(self):
        # Setup PitchYin with specific parameters
        return es.PitchYin(sampleRate=44100,
                           frameSize=2048,
                           maxFrequency=22050,
                           minFrequency=20,
                           interpolate=True,
                           tolerance=0.15)
    
    def setupPlot(self):
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Real-Time Pitch Tracking')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Pitch (Hz)')
        self.line, = self.ax.plot([], [], 'r-')
        self.ax.set_xlim(0, 10)  # Set this to your expected time window
        self.ax.set_ylim(20, 22050)  # Pitch range from minFrequency to maxFrequency
        self.times = []
        self.pitches = []

    def updatePlot(self, pitch, confidence):
        # Update plot with new data
        self.times.append(self.times[-1] + 0.1 if self.times else 0)  # Assuming a frame every 0.1 seconds
        self.pitches.append(pitch)
        self.line.set_data(self.times, self.pitches)
        self.ax.set_xlim(min(self.times), max(self.times) + 1)
        self.canvas.draw()

    def toggleRecording(self):
        # Toggle the recording state
        if self.is_recording:
            self.is_recording = False
            self.button.setText('Start Recording')
            self.stream.stop_stream()
        else:
            self.is_recording = True
            self.button.setText('Stop Recording')
            self.stream.start_stream()

    def callback(self, in_data, frame_count, time_info, status):
        # Audio stream callback function
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.printPitchYin(audio_data)
            # print(f'RMS: {np.sqrt(np.mean(audio_data**2))}')  # Calculate and print RMS amplitude
        return (in_data, pyaudio.paContinue)

    def printPitchYin(self, audio_data):
        # Normalize the buffer from int16 range to floating-point range
        audio_data_float = audio_data.astype(np.float32) / 32768.0
        pitch, pitchConfidence = self.pitchYin(audio_data_float)
        print(f"Estimated pitch: {pitch} Hz, Confidence: {pitchConfidence}")
        self.updatePlot(pitch, pitchConfidence)

    def closeEvent(self, event):
        # Handle the close event
        self.stream.close()
        self.audio.terminate()


if __name__ == '__main__':
    ARgui = AudioRecorder()
    ARgui.run()