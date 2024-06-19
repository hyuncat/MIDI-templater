import sys
import wave
import simpleaudio as sa
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer, Qt

class AudioPlayer(QMainWindow):
    def __init__(self, audio_file):
        super().__init__()
        self.audio_file = audio_file
        self.play_obj = None
        self.is_playing = False
        self.start_time = 0
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Simple Audio Player")
        self.setGeometry(100, 100, 300, 100)
        
        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        
        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.seek)
        layout.addWidget(self.slider)
        
        widget.setLayout(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_slider)

    def load_audio(self):
        with wave.open(self.audio_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            self.audio_length = frames / rate * 1000  # Length in milliseconds
        
        self.wave_obj = sa.WaveObject.from_wave_file(self.audio_file)
        self.slider.setMaximum(int(self.audio_length))

    def toggle_playback(self):
        if self.is_playing:
            self.play_button.setText('Play')
            self.play_obj.stop()
            self.timer.stop()
            self.is_playing = False
        else:
            self.play_button.setText('Pause')
            self.play_obj = self.wave_obj.play()
            self.start_time = self.slider.value()  # Set start time from slider
            self.timer.start(100)
            self.is_playing = True

    def update_slider(self):
        if self.is_playing:
            elapsed_time = self.start_time + self.timer.interval()  # Increase elapsed time by timer interval
            self.slider.setValue(elapsed_time)
            self.start_time = elapsed_time  # Update start time for the next tick

    def seek(self, value):
        if self.is_playing:
            self.play_obj.stop()
            self.play_obj = self.wave_obj.play(start=value / 1000)
            self.start_time = value

    def closeEvent(self, event):
        if self.play_obj:
            self.play_obj.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = AudioPlayer('../data/mozart_vc4_mvt1.wav')
    player.load_audio()
    player.show()
    sys.exit(app.exec())
