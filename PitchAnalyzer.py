import sys
import numpy as np
import pyaudio
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QMainWindow, QLabel
import essentia.standard as es
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class PitchAnalyzer(QMainWindow):
    def __init__(self, argv=sys.argv):
        self.app = QApplication(argv)
        super().__init__()
        self.setupPlot()
        self.initUI()
        self.initAudio()
        self.pitchYin = self.ES_PitchYin()
        self.is_recording = False

    def run(self):
        self.show()
        sys.exit(self.app.exec_())

    def initUI(self):
        # Set up the main window and central widget
        self.setWindowTitle('Pitch Analyzer')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Set up the GUI layout with a button
        self.button = QPushButton('Start Recording', self)
        self.button.clicked.connect(self.toggleRecording)
        layout.addWidget(self.button)

        # Plot setup
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Optionally, add a status bar
        self.statusBar().showMessage('Ready')

    def initAudio(self):
        # Set up audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=44100,
                                      input=True,
                                      frames_per_buffer=1024,
                                      stream_callback=self.callback)

    def ES_PitchYin(self):
        # Initialize PitchYin with specific parameters
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
        self.ax.set_xlabel('Time (milliseconds)')
        self.ax.set_ylabel('Pitch (Hz)')
        self.line, = self.ax.plot([], [], 'r-')
        self.ax.set_xlim(0, 5000)  # 5 seconds expressed in milliseconds
        self.ax.set_ylim(20, 22050)  # Pitch range
        self.times = []
        self.pitches = []

    def updatePlot(self, pitch, confidence):
        time_step = 100  # time step in milliseconds
        self.times.append(self.times[-1] + time_step if self.times else 0)
        self.pitches.append(pitch)
        
        # Keep only the last 5 seconds of data
        if self.times[-1] > 5000:
            min_time = self.times[-1] - 5000
            self.times = [t - min_time for t in self.times]
            self.pitches = self.pitches[-len(self.times):]

        self.line.set_data(self.times, self.pitches)
        self.ax.set_xlim(self.times[0], self.times[-1])
        self.canvas.draw()

    def toggleRecording(self):
        # Toggle the recording state
        if self.is_recording:
            self.is_recording = False
            self.button.setText('Start Recording')
            self.stream.stop_stream()
            self.statusBar().showMessage('Stopped recording')
        else:
            self.is_recording = True
            self.button.setText('Stop Recording')
            self.stream.start_stream()
            self.statusBar().showMessage('Recording...')

    def callback(self, in_data, frame_count, time_info, status):
        # Audio stream callback function
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.printPitchYin(audio_data)
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
    PitchGUI = PitchAnalyzer()
    PitchGUI.run()
