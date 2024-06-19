import sys
import os
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QApplication, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont 
import qdarktheme
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pygame

import pretty_midi
from midi2audio import FluidSynth
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa
import pyaudio
import essentia.standard as es

class MidiPlotter(QMainWindow):
    def __init__(self, midi_file):
        super().__init__()  # Call the parent class (QMainWindow) constructor
        pygame.mixer.init()  # Initialize the pygame mixer

        self.midi_file = midi_file
        self.soundfont = 'data/MuseScore_General.sf3'
        self.FS = FluidSynth(self.soundfont)
        
        self.instrument_dfs = self.load_midi_data(midi_file)

        self.sample_rate = 44100
        self.plot_buffer = np.array([])
        self.plot_buffer_size = 4410
        self.pitchData = {} # Dictionary to store pitch data tuples (time, pitch)
        self.initAudio()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_slider_with_timer)
        self.current_time = 0 # in sec
        self.current_slider = 0 # in ms
        self.is_playing = False
        
        self.initUI()
        self.pitchYin = self.ES_PitchYin()
        

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
        """Initialize the UI with the play button, slider, and plot widget"""
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Set title
        self.setWindowTitle('Real-Time Pitch Correction')

        # Set heading label
        self.headingLabel = QLabel('Real-Time Pitch Correction', self)
        self.headingLabel.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        # Set heading font
        font = QFont()
        font.setPointSize(18)  # Set the font size to 18 points, adjust as needed
        self.headingLabel.setFont(font)
        self.layout.addWidget(self.headingLabel)  # Add the label to the layout 

        self.plotWidget = pg.PlotWidget()
        self.layout.addWidget(self.plotWidget)
        self.plot_midi_data()

        # Add slider
        self.initSlider()

        # Create a horizontal layout for buttons
        self.buttonLayout = QHBoxLayout()

        # Play Button
        self.playButton = QPushButton('Play', self)
        self.playButton.clicked.connect(self.toggle_play)
        self.buttonLayout.addWidget(self.playButton)

        # Reset Button
        self.resetButton = QPushButton('Reset recording', self)
        self.resetButton.clicked.connect(self.clear_pitch_data)
        self.buttonLayout.addWidget(self.resetButton)

        # Add the button layout to the main layout
        self.layout.addLayout(self.buttonLayout)


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
        """Plot the MIDI data on the plot widget"""
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
            brush='b',
            name='MIDI')
        self.plotWidget.addItem(self.bars)

        # The current time line (red)
        self.currentTimeLine = pg.InfiniteLine(
            pos=self.current_time, 
            angle=90, 
            pen={'color': 'r', 'width': 2})
        self.plotWidget.addItem(self.currentTimeLine)

        # Plot pitch data as scatter points
        times = list(self.pitchData.keys())
        pitches = [self.pitchData[time] for time in times]
        
        # Normalize pitch data (you might need to adjust this formula based on your specific pitch range)
        normalized_pitches = [(12*np.log2(pitch/440)+69) for pitch in pitches]

        print(f'Plotting pitches: {normalized_pitches}')
        self.pitchPlot = pg.PlotCurveItem(
            x=times, 
            y=normalized_pitches, 
            pen={'color': 'r', 'width': 2},
            name='Recorded Pitch')
        self.plotWidget.addItem(self.pitchPlot)

        self.legend = pg.LegendItem(offset=(55, 10))
        self.legend.setParentItem(self.plotWidget.graphicsItem())
        self.legend.addItem(self.bars, 'MIDI')
        self.legend.addItem(self.pitchPlot, 'Recorded Pitch')

        self.plotWidget.setLabel('bottom', 'Time (s)')  # X-axis label
        self.plotWidget.setLabel('left', 'Pitch (MIDI #)')  # Y-axis label
        
        # Adjust plot margins
        self.plotWidget.setXRange(self.current_time - 5, self.current_time + 5, padding=0.1)
        

    def toggle_play(self):
        """Toggle between playing and pausing the MIDI."""
        if self.is_playing:
            pygame.mixer.music.set_pos(self.current_time)
            pygame.mixer.music.pause() # Pause music
            self.is_playing = False
            self.playButton.setText('Play')
            self.timer.stop() # Stop updating the slider
            self.stream.stop_stream() # Stops recording
            print('Pausing')
        else:
            pygame.mixer.music.play()
            pygame.mixer.music.set_pos(self.current_time)
            self.is_playing = True
            self.playButton.setText('Pause')
            self.timer.start(100)  # Update every 100 ms
            self.stream.start_stream()
            print('Playing')
        

    def slider_moved(self, position):
        """Called when user moves the slider to [position]"""
        self.current_slider = position
        self.current_time = self.current_slider / 10
        self.slider.setValue(int(self.current_slider))
        pygame.mixer.music.set_pos(self.slider.value() / 10)
        self.update_plot()

    def slider_changed(self):
        """Called whenever slider's value changes"""
        self.update_plot()

    def update_slider_with_timer(self):
        """Called at each self.timer.timeout.connect() to update the slider value"""
        print(f'Old slider: {self.current_slider}')
        if self.is_playing:
            self.current_slider += 1 # where 1 tick == 100 ms
            self.current_time = self.current_slider / 10
            self.slider.setValue(int(self.current_slider))
            print(f'Updating slider to {self.current_time} sec')
            self.update_plot()

    def update_plot(self):
        """Updates the plot to reflect the current time based on the slider's position."""
        self.plotWidget.setXRange(self.current_time - 5, self.current_time + 5, padding=0.1)
        self.currentTimeLine.setPos(self.current_time)

        if self.pitchData:
            # Plot pitch data as scatter points
            times = list(self.pitchData.keys())
            pitches = [self.pitchData[time] for time in times]
            
            # Normalize pitch data (you might need to adjust this formula based on your specific pitch range)
            normalized_pitches = [(12*np.log2(pitch/440)+69) for pitch in pitches]

            # print(f'Plotting pitches: {normalized_pitches}')
            if hasattr(self, 'pitchPlot'):
                self.plotWidget.removeItem(self.pitchPlot)
            self.pitchPlot = pg.PlotCurveItem(
                x=times, 
                y=normalized_pitches, 
                pen={'color': 'r', 'width': 2})
            self.plotWidget.addItem(self.pitchPlot)
        else:
            self.plotWidget.removeItem(self.pitchPlot)

    def clear_pitch_data(self):
        """Clears the pitch data and updates the plot."""
        self.pitchData.clear()  # Reset the pitch data dictionary
        self.update_plot()  # Update the plot to reflect the cleared data
        

    def stop_midi(self):
        pygame.mixer.music.stop()
        self.timer.stop()
        self.playButton.setText('Play')
        self.is_playing

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
                           maxFrequency=5000,
                           minFrequency=150,
                           sampleRate=44100,
                           tolerance=0.15)
    

    def callback(self, in_data, frame_count, time_info, status):
        """
        Overloaded method for pyaudio to call, handles audio stream callback
        """
        if self.is_playing:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.plot_buffer = np.append(self.plot_buffer, audio_data)
            self.printPitchYin(audio_data)
        return (in_data, pyaudio.paContinue)

    def printPitchYin(self, audio_data):
        # Normalize the buffer from int16 range to floating-point range
        audio_data_float = audio_data.astype(np.float32) / 32768.0
        pitch, pitchConfidence = self.pitchYin(audio_data_float)
        
        if pitchConfidence > 0.5:  # Only plot pitches with high confidence
            # print(f"Time: {self.current_time} Pitch: {pitch} Midi: {(12*np.log2(pitch/440)+69)} Hz, Confidence: {pitchConfidence}")
            self.pitchData[self.current_time] = pitch

        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    midi_file_path = 'data/mozart_vc4_mvt1.mid' 
    ex = MidiPlotter(midi_file_path)
    ex.show()
    sys.exit(app.exec())
