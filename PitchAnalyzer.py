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
        """
        Initialize PitchAnalyzer GUI instance
        """
        self.app = QApplication(argv)
        super().__init__()

        # Setup the PitchPlot (maybe other plots later)
        self.figure = Figure()
        self.setupPitchPlot()

        # Initialize the UI
        self.initPitchUI()

        # Initialize audio
        self.initAudio()
        self.is_recording = False
        self.buffer = np.array([])
        self.buffer_size = 4410  # 100ms buffer

        self.pitchYin = self.ES_PitchYin()

    def run(self):
        """
        Self-contained method to run the GUI
        """
        self.show()
        sys.exit(self.app.exec_())

    def initPitchUI(self):
        # Main window and central widget
        self.setWindowTitle('Pitch Analyzer')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        self.statusBar().showMessage('Ready')

        # Recording start/stop button!
        self.button = QPushButton('Start Recording', self)
        self.button.clicked.connect(self.toggleRecording)
        layout.addWidget(self.button)

        # Plot setup
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)


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
        """
        Initialize Essentia's PitchYin algorithm
        """
        #TODO: make parameters configurable
        return es.PitchYin(frameSize=2048,
                           interpolate=True,
                           maxFrequency=22050,
                           minFrequency=20,
                           sampleRate=44100,
                           tolerance=0.15)

    def setupPitchPlot(self):
        """
        Setup pitch plot
        """
        # Add axes
        self.ax = self.figure.add_subplot(111) #1row x 1col x 1st subplot
        self.ax.set_title('Real-Time Pitch Tracking')
        self.ax.set_xlabel('Time (milliseconds)')
        self.ax.set_ylabel('Pitch (Hz)')

        self.ax.set_xlim(0, 5000)  # 5 sec -> ms
        self.ax.set_ylim(20, 22050)  # Pitch range

        # Stores the pitch line
        self.line, = self.ax.plot([], [], 'r-') # TODO: account for confidence
        self.times = []
        self.pitches = []
        

    def updatePitchPlot(self, pitch, confidence):
        """
        Update parameters for the pitch plot and draws the new canvas
        @param: 
            - pitch: estimated pitch in Hz
            - confidence: confidence of the pitch estimation
        note: both param are return values from es.pitchYin()
        """
        time_step = 100  # timestep (ms)
        
        # Update times and pitches for plotting
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
        """
        Toggle application to start/stop recording when button is clicked
        """
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
        """
        Overloaded method for pyaudio to call, handles audio stream callback
        """
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.printPitchYin(audio_data)
        return (in_data, pyaudio.paContinue)

    def printPitchYin(self, audio_data):
        # Normalize the buffer from int16 range to floating-point range
        audio_data_float = audio_data.astype(np.float32) / 32768.0
        pitch, pitchConfidence = self.pitchYin(audio_data_float)
        print(f"Estimated pitch: {pitch} Hz, Confidence: {pitchConfidence}")
        self.updatePitchPlot(pitch, pitchConfidence)

    def closeEvent(self, event):
        # Handle the close event
        self.stream.close()
        self.audio.terminate()


if __name__ == '__main__':
    PitchGUI = PitchAnalyzer()
    PitchGUI.run()
