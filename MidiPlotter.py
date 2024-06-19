import sys
import pretty_midi
import pandas as pd
import os

from PyQt6.QtWidgets import QMainWindow, QApplication, QSlider, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
import pygame

from midi2audio import FluidSynth
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa

class MidiPlotter(QMainWindow):
    def __init__(self, midi_file):
        super().__init__()  # Call the parent class (QMainWindow) constructor
        pygame.mixer.init()  # Initialize the pygame mixer

        self.midi_file = midi_file
        self.soundfont = 'MuseScore_General.sf3'
        self.FS = FluidSynth(self.soundfont)
        
        self.instrument_dfs = self.load_midi_data(midi_file)
        self.initUI()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_slider_with_timer)
        self.current_time = 0 # in sec
        self.current_slider = 0 # in ms
        self.is_playing = False

        # self.play_obj = None # simpleaudio play object

    def load_midi_data(self, midi_file):
        """Load the MIDI data and convert to a pd.DataFrame for each instrument"""
        wav_filename = midi_file.replace('.mid', '.wav')
        if not os.path.exists(wav_filename):
            self.FS.midi_to_audio(midi_file, wav_filename)

        self.midi_data = pretty_midi.PrettyMIDI(midi_file)
        pygame.mixer.music.load(wav_filename)

        instrument_dfs = {}
        for instrument in self.midi_data.instruments:
            rows = []
            for note in instrument.notes:
                row = [note.pitch, note.get_duration(), note.start, note.end]
                rows.append(row)
            df = pd.DataFrame(rows, columns=['pitch', 'duration', 'start', 'end'])
            instrument_dfs[instrument.program] = df
        return instrument_dfs

    def initUI(self):
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.playButton = QPushButton('Play', self)
        self.playButton.clicked.connect(self.toggle_play)
        self.layout.addWidget(self.playButton)

        self.initSlider()

        self.plotWidget = pg.PlotWidget()
        self.layout.addWidget(self.plotWidget)
        self.plot_midi_data()

    def initSlider(self):
        """Initialize the slider with min/max values and its connected signals"""
        self.slider = QSlider(Qt.Orientation.Horizontal) # The QSlider object
        
        # Set slider range + initial value
        self.slider.setValue(0)
        self.slider.setMinimum(0)
        midi_length_seconds = int(self.midi_data.get_end_time())
        self.slider.setMaximum(midi_length_seconds*10)

        # Connect slider signals
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.valueChanged.connect(self.slider_changed)
        self.layout.addWidget(self.slider)


    def plot_midi_data(self):
        self.plotWidget.clear()
        self.violin_df = self.instrument_dfs[40]
        self.bar_height = 1  # Fixed height for all bars
        # Calculate the bottom position of each bar so the top aligns with the pitch
        y_positions = self.violin_df['pitch'] - self.bar_height
        self.bars = pg.BarGraphItem(
            x=self.violin_df['start'], 
            y=y_positions,  # Adjusted y position
            height=self.bar_height,  # Fixed height
            width=self.violin_df['duration'], 
            brush='b')
        self.plotWidget.addItem(self.bars)

    def toggle_play(self):
        if self.is_playing:
            pygame.mixer.music.set_pos(self.current_time)
            pygame.mixer.music.pause()
            self.is_playing = False
            self.playButton.setText('Play')
            self.timer.stop()
            print('Pausing')
        else:
            pygame.mixer.music.play()
            pygame.mixer.music.set_pos(self.current_time)
            self.is_playing = True
            self.timer.start(100)  # Update every 100 ms

    def slider_moved(self, position):
        """Called when user moves the slider to [position]"""
        self.current_slider = position
        self.current_time = self.current_slider / 10
        self.slider.setValue(int(self.current_slider))
        self.update_plot()

    def slider_changed(self):
        """Called whenever slider's value changes"""
        if self.is_playing:
            pygame.mixer.music.set_pos(self.slider.value())
            self.current_time = pygame.mixer.music.get_pos()
        self.current_slider = self.current_time * 10
        self.slider.setValue(int(self.current_slider))
        self.update_plot()

    def update_slider_with_timer(self):
        """Called at each self.timer.timeout.connect() to update the slider value"""
        if self.is_playing:
            self.current_slider += 1 # where 1 tick == 100 ms
            self.current_time = self.current_slider / 10
            self.slider.setValue(int(self.current_slider))
            self.update_plot()

    def update_plot(self):
        """Updates the plot to reflect the current time based on the slider's position."""
        self.plotWidget.setXRange(self.current_time - 5, self.current_time + 5, padding=0.1)

    def stop_midi(self):
        pygame.mixer.music.stop()
        self.timer.stop()
        self.playButton.setText('Play')
        self.is_playing


if __name__ == '__main__':
    app = QApplication(sys.argv)
    midi_file_path = 'mozart_vc4_mvt1.mid'  # Replace with your actual MIDI file path
    ex = MidiPlotter(midi_file_path)
    ex.show()
    sys.exit(app.exec())
