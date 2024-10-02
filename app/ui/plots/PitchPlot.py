import os
import json
import numpy as np
import pandas as pd

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt6.QtGui import QColor

from app.modules.midi.MidiData import MidiData
from app.modules.pitch.Pitch import Pitch


class PitchPlot(QWidget):
    def __init__(self):
        super().__init__()

        self.colors = {
            'background': '#1D1D1D', # Darkest grey
            'label_text': '#A4A4A4',  # Black
            'timeline': '#FF0000',  # Red color
            'staff_line': '#363636',  # Dark grey
            'onsets': '#C297A2', # Rosy brown
            'notes': '#8B768A' # Mountbatten pink
        }
        self.midi_config = {
            'normal_color': '#53555C',  # Grey
            'played_color': '#3A3B40',  # Darker grey
            'bar_height': 1
        }
        self.user_config = {
            'pitch_color': '#F57C9C',  # Pink
        }
        # self.staff_lines = {
        #     'E4': 64,
        #     'G4': 67,
        #     'B4': 71,
        #     'D5': 74,
        #     'F5': 77
        # }

        self.current_time = 0
        self.x_range = 5
        self.y_bounds = (40, 90) 
        self.timeline_xpos = 1/5 # x-pos of the timeline in the plot

        self.init_ui()
        self.add_staff_lines()
        self.init_timeline()
    
    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.plot = pg.PlotWidget()
        self.layout.addWidget(self.plot)

        self.plot.setBackground(self.colors['background'])
        
        self.plot.getAxis('bottom').setPen(self.colors['label_text'])
        self.plot.getAxis('left').setPen(self.colors['label_text'])
        self.plot.getAxis('bottom').setTextPen(self.colors['label_text'])
        self.plot.getAxis('left').setTextPen(self.colors['label_text'])

        # Add axis labels
        self.plot.setLabel('bottom', 'Time (s)', color=self.colors['label_text'])  # X-axis label
        self.plot.setLabel('left', 'Pitch (MIDI #)', color=self.colors['label_text'])  # Y-axis label

        # Set padding for plot area to create space for labels
        self.plot.setContentsMargins(10, 10, 10, 10)  # Margins around the widget itself
        self.plot.getPlotItem().layout.setContentsMargins(10, 10, 10, 20)  # Margins inside the plot layout

        # Set the range of the plot
        x_lower = self.current_time - (self.timeline_xpos * self.x_range)
        x_upper = self.current_time + ((1 - self.timeline_xpos) * self.x_range)
        self.plot.setXRange(x_lower, x_upper)
        self.plot.setYRange(self.y_bounds[0], self.y_bounds[1])
        
    def add_staff_lines(self):
        """Add staff lines to the plot to mimic the treble clef staff."""

        # Load the staff lines from the JSON file
        app_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        staff_lines_filepath = os.path.join(app_directory, 'resources', 'pitch_json.json')
        with open(staff_lines_filepath, 'r') as file:
            self.all_staff_lines = json.load(file)

        # Add the staff lines to the plot
        for midi_number in self.all_staff_lines.values():
            line = pg.InfiniteLine(pos=midi_number, angle=0, pen={'color': self.colors['staff_line'], 'width': 1})
            self.plot.addItem(line)

        # Add y-axis ticks for each staff line
        y_axis = self.plot.getAxis('left')
        ticks = [(midi_number, note) for note, midi_number in self.all_staff_lines.items()]
        y_axis.setTicks([ticks])
        y_axis.setStyle(tickTextOffset=10, tickFont=pg.QtGui.QFont('Arial', 8))

    
    def init_timeline(self):
        """Add a vertical line to the plot to indicate the current time."""
        self.current_timeline = pg.InfiniteLine(
            pos=self.current_time,
            angle=90,
            pen={'color': self.colors['timeline'], 'width': 2})
        self.plot.addItem(self.current_timeline)

    def plot_midi(self, midi_data: MidiData):
        """Plot the MIDI data on the pitch plot."""

        if midi_data is not None:
            self.midi_data = midi_data
        elif self.midi_data is None and midi_data is None:
            raise ValueError('No MIDI data to plot.')

        pitch_df = self.midi_data.pitch_df

        self.midi_notes = []
        for start, duration, pitch in zip(pitch_df['start'], pitch_df['duration'], pitch_df['pitch']):
            note_x = start + (duration / 2)
            color = self.midi_config['normal_color'] if note_x >= self.current_time else self.midi_config['played_color']

            midi_note = pg.BarGraphItem(
                x=note_x,  
                y=pitch,
                height=self.midi_config['bar_height'],
                width=duration,
                brush=color,
                pen=None,
                name='MIDI')
            self.midi_notes.append(midi_note)
            self.plot.addItem(midi_note)


    def plot_pitches(self, pitches: list[Pitch]):
        """Plot the detected pitches on the pitch plot."""

        print("Plotting pitches...")
        base_color = QColor(self.user_config['pitch_color'])  # Pink
        # freq_to_midi = lambda freq: 12*np.log(freq/220)/np.log(2) + 57

        # Step 1: Normalize volumes (min-max normalization)
        volumes = [pitch.volume for pitch in pitches]
        min_volume = min(volumes)
        max_volume = max(volumes)
        
        # Avoid division by zero in case all volumes are the same
        if max_volume == min_volume:
            normalized_volumes = [0.5] * len(pitches)  # Set all to mid-range if no variation in volume
        else:
            normalized_volumes = [(v - min_volume) / (max_volume - min_volume) for v in volumes]


        brushes = []
        for i, pitch in enumerate(pitches):
            # color = QColor(base_color)  # Start with the base color
            # # color.setAlphaF(pitch.probability)  # Set the alpha based on pitch probability (0 to 1)
            # color.setAlphaF(pitch.volume)  # Set the alpha based on pitch probability (0 to 1)
            # brushes.append(pg.mkBrush(color))  # Use mkBrush to create the brush with color

            # Set hue based on volume: blue (quiet) -> red (medium) -> yellow (loud)
            # Volume is assumed to be in range [0, 1]
            volume = normalized_volumes[i]

            # Convert volume to hue:
            # 0 -> blue (HSV hue ~240), 0.5 -> red (HSV hue ~0), 1 -> yellow (HSV hue ~60)
            if volume < 0.5:
                hue = 240 - (240-0) * (volume/0.5)  # blue to red
            else:
                hue = 0 + (60 - 0) * ((volume-0.5) / 0.5)  # red to yellow

            # QColor uses HSV (hue, saturation, value) to represent color
            color = QColor.fromHsv(int(hue), 255, 255)  # Full saturation and value for bright colors

            # Set opacity based on pitch probability
            color.setAlphaF(pitch.probability)  # Opacity (alpha) based on pitch probability (0 to 1)
            brushes.append(pg.mkBrush(color))  # Use mkBrush to create the brush with color


        self.pitches = pg.ScatterPlotItem(
            x=[pitch.time for pitch in pitches],
            y=[pitch.midi_num for pitch in pitches],
            size=5,
            pen=None,
            brush=brushes,
            name='Pitch'
        )
        self.plot.addItem(self.pitches)
        print("Done!")

    def plot_notes(self, note_df: pd.DataFrame):
        """Plot the detected notes on the pitch plot."""

        self.note_lines = []
        if len(note_df) > 1:
            for j in range(len(note_df) - 1):
                note_line = pg.PlotCurveItem(
                    x=[note_df.iloc[j]['time'], note_df.iloc[j + 1]['time']],
                    y=[note_df.iloc[j]['midi_num'], note_df.iloc[j]['midi_num']],
                    pen=pg.mkPen(self.colors['notes'], width=25),
                    name='Notes'
                )
                self.plot.addItem(note_line)
                self.note_lines.append(note_line)


    def plot_onsets(self, onsets: np.ndarray):
        """Plot the detected onsets on the pitch plot."""
        
        self.onsets = []
        for onset in onsets:
            onset_bar = pg.BarGraphItem(
                x=onset,
                y=71,
                height=5,
                width=0.005,
                brush=self.colors['onsets'],
                pen=None,
                name='Onset'
            )
            self.onsets.append(onset_bar)
            self.plot.addItem(onset_bar)

    def move_plot(self, current_time: float):
        """Move the plot to the current time."""
        self.current_time = current_time
        self.current_timeline.setPos(self.current_time)

        # Set the range of the plot
        x_lower = self.current_time - (self.timeline_xpos * self.x_range)
        x_upper = self.current_time + ((1 - self.timeline_xpos) * self.x_range)
        self.plot.setXRange(x_lower, x_upper)

        # Update the color of the MIDI notes based on the current time
        for midi_note in self.midi_notes:
            color = self.midi_config['normal_color'] if midi_note.opts['x'] >= self.current_time else self.midi_config['played_color']
            midi_note.setOpts(brush=color)


class RunPitchPlot:
    def __init__(self, app=None, midi_data: MidiData=None, pitches: list[Pitch]=None, onsets: np.ndarray=None, note_df: pd.DataFrame=None):
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from app.config import AppConfig
        from PyQt6.QtWidgets import QApplication, QMainWindow, QToolBar
        from PyQt6.QtCore import Qt

        if app is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = app
        
        AppConfig.initialize(self.app)
        self.mainWindow = QMainWindow()
        self.mainWindow.setWindowTitle("Pitch Plot")
        self.mainWindow.setGeometry(100, 100, 800, 600)

        # Create a central widget and set the layout for it
        self.centralWidget = QWidget()
        self.mainLayout = QVBoxLayout(self.centralWidget)
        self.mainWindow.setCentralWidget(self.centralWidget)

        # Initialize VisualizeWindow and add it to the layout
        self.pitch_plot = PitchPlot()
        self.mainLayout.addWidget(self.pitch_plot)

        # Create and configure the toolbar
        self.toolbar = QToolBar("Main Toolbar", self.mainWindow)
        self.toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.toolbar.addAction("Exit", self.close)
        self.mainWindow.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Load midi if provided
        if midi_data is not None:
            self.pitch_plot.plot_midi(midi_data)
        if onsets is not None:
            self.pitch_plot.plot_onsets(onsets)
        if note_df is not None:
            self.pitch_plot.plot_notes(note_df)
        if pitches is not None:
            self.pitch_plot.plot_pitches(pitches)

        self.mainWindow.show()
        self.app.exec()

    def close(self):
        self.app.quit()
    
