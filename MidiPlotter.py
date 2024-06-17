import sys
import pretty_midi
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QApplication, QSlider, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import pygame

class MidiPlotter(QMainWindow):
    def __init__(self, midi_file):
        super().__init__() # Call the parent class (QMainWindow) constructor
        pygame.mixer.init() # Initialize the pygame mixer

        # Load in a specific MIDI file (later implement file IO)
        self.midi_file = midi_file
        self.instrument_dfs = self.load_midi_data(midi_file)

        # Initialize the GUI
        self.initUI()

        # Timer variables to update internal state/playback with slider
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot_during_playback)
        self.current_time = 0
        self.is_playing = False


    def load_midi_data(self, midi_file):
        """
        Load in a MIDI file and convert the violin instrument into a dataframe of 
        pitches, durations, start, and end times for all violin notes.
        """
        self.midi_data = pretty_midi.PrettyMIDI(midi_file)
        pygame.mixer.music.load(midi_file)

        # Get instrument_dfs of all instruments in MIDI
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
        """
        Initialize the GUI - setup main window, buttons, plot
        """
        # Set up the main window
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Play button
        self.playButton = QPushButton('Play', self)
        self.playButton.clicked.connect(self.toggle_play)
        layout.addWidget(self.playButton)

        # Slider to control playback position
        self.slider = QSlider(Qt.Horizontal)

        # Set the slider range to the length of the MIDI file
        midi_length_seconds = int(self.midi_data.get_end_time())
        self.slider.setMaximum(midi_length_seconds)
        self.slider.setValue(0)

        # Connect the slider to the playback position
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.valueChanged.connect(self.update_plot_during_playback)
        layout.addWidget(self.slider)

        # Plot widget
        self.plotWidget = pg.PlotWidget()
        layout.addWidget(self.plotWidget)
        
        self.plot_midi_data() # Plot the MIDI data


    def plot_midi_data(self):
        """
        Plot the MIDI data as a bar graph of pitches.
        (Probably change to a line graph later)
        """

        self.plotWidget.clear()
        violin_df = self.instrument_dfs[40]  # Violin instrument is program 40

        # Use bar graph for notes; 'stepMode=True' makes bars have equal width
        self.bars = pg.BarGraphItem(
            x=violin_df['start'], 
            height=violin_df['pitch'], 
            width=violin_df['duration'], 
            brush='b')
        self.plotWidget.addItem(self.bars)


    def toggle_play(self):
        print(f'toggling play - is_playing: {self.is_playing}')
        if self.is_playing:
            print("pausing playback")
            pygame.mixer.music.stop()
            self.is_playing = False
            self.playButton.setText('Play')
        else:
            if pygame.mixer.music.get_pos() == -1 or pygame.mixer.music.get_pos() == 0:
                print("starting playback")
                print(f"current_time: {self.current_time}")
                pygame.mixer.music.play()
                print("starting playback")
            else:
                print("starting playback")
                print(f"current_time: {self.current_time}")
                pygame.mixer.music.set_pos(self.current_time)
                pygame.mixer.music.play()
                print("unpausing playback")
                # pygame.mixer.music.unpause()
            self.is_playing = True
            self.playButton.setText('Pause')
            self.timer.start(100)  # Update every 100 ms


    def slider_moved(self, position):
        self.current_time = position
        pygame.mixer.music.set_pos(position)
        self.plotWidget.setXRange(self.current_time-5, self.current_time+5, padding=0.1)


    def update_plot_during_playback(self):
        if pygame.mixer.music.get_busy():
            self.current_time = pygame.mixer.music.get_pos() / 1000.0  # Convert milliseconds to seconds
            self.slider.setValue(int(self.current_time))  # Update slider position
            self.plotWidget.setXRange(self.current_time - 5, self.current_time + 5, padding=0.1)

    # def update_plot_during_playback(self):
    #     if pygame.mixer.music.get_busy():
    #         # Calculate the current playback position
    #         self.current_time = pygame.mixer.music.get_pos() / 1000.0  # get_pos returns milliseconds
    #         self.slider.setValue(self.df[self.df['start'] < self.current_time].index[-1])
    #         # Update plot to reflect current playback position
    #         self.plotWidget.setXRange(self.current_time - 5, self.current_time + 5, padding=0.1)

    def stop_midi(self):
        pygame.mixer.music.stop()
        self.timer.stop()
        self.playButton.setText('Play')
        self.is_playing = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    midi_file_path = 'mozart_vc4_mvt1.mid'  # Replace with your actual MIDI file path
    ex = MidiPlotter(midi_file_path)
    ex.show()
    sys.exit(app.exec_())
