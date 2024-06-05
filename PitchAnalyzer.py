import sys
import numpy as np
import pyaudio
import essentia.standard as es

from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QMainWindow, QLabel
import qdarktheme

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
        self.app.setStyleSheet(qdarktheme.load_stylesheet("light"))

        # Setup the PitchPlot (maybe other plots later)
        self.figure = Figure()
        self.setupPitchPlot()

        # Initialize the UI
        self.initPitchUI()

        # Initialize audio
        self.sample_rate = 44100 # 44.1 kHz == 44100 samples/sec (CD standard)
        self.initAudio()
        self.is_recording = False

        # Plot update buffer variables
        self.plot_buffer = np.array([])
        self.plot_buffer_size = 4410  # 100ms buffer
        self.current_time = 0

        self.pitchYin = self.ES_PitchYin()


    def run(self):
        """
        Self-contained method to run the GUI
        """
        self.show()
        sys.exit(self.app.exec_())


    def initPitchUI(self):
        """
        Draw the PitchAnalyzer GUI onto the main window
        """
        # Main window title
        self.setWindowTitle('RT-PitchAnalyzer')

        # Central widget 'layout' contains all other buttons/plots
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
        """
        Initialize PyAudio input stream
        Note: Frames per buffer is 1024 samples/buffer /44100 samples/sec = .023 buffer/sec (?)
        """
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            output=False,
            frames_per_buffer=1024, # 23ms buffer (audio processing standard)
            stream_callback=self.callback
        )


    def ES_PitchYin(self):
        """
        Initialize Essentia PitchYin algorithm
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
        Setup pitch plot with Matplotlib
        """
        # Add axes
        self.ax = self.figure.add_subplot(111) #1row x 1col x 1st subplot
        self.ax.set_title('Real-Time Pitch Tracking')
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Pitch (Hz)')

        self.ax.set_xlim(0, 5)  # 5 sec -> ms
        self.ax.set_ylim(20, 3000)  # Pitch range

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
        time_step = 0.1  # timestep (100 ms)
        
        if confidence < 0.05:
            print("Confidence too low, skipping plot")
            return

        # Update times and pitches for plotting
        self.times.append(self.times[-1] + time_step if self.times else 0)
        self.pitches.append(pitch)
        
        # Update x-axis limits to show only the last 5 seconds of data
        if self.times[-1] > 5:
            min_time = self.times[-1] - 5
            max_time = self.times[-1]
            self.ax.set_xlim(min_time, max_time)

        self.line.set_data(self.times, self.pitches)
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
            self.plot_buffer = np.append(self.plot_buffer, audio_data)
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
