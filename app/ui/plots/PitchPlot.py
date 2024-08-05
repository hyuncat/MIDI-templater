from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app.modules.midi.MidiData import MidiData
from app.modules.pitch.PitchAnalyzer import PitchAnalyzer

class PitchPlot(QWidget):
    def __init__(self):
        super().__init__()

        self.colors = {
            'background': '#ECECEC',
            'label_text': '#000000',  # Black color
            'MIDI_normal': '#000000',  # Black
            'MIDI_darker': '#377333',  # Green
            'user_note': '#B66CD3',  # Lavendar
            'timeline': '#FF0000',  # Red color
            'staff_line': '#A9A9A9',  # Dark Gray for staff lines
            'onset': '#B1B1B1'  # Light blue for onsets
        }

        self.staff_lines = {
            'E4': 64,
            'G4': 67,
            'B4': 71,
            'D5': 74,
            'F5': 77
        }

        self.init_ui()
        self.x_range = 3  # Show 3 seconds of pitch data at once
        self.current_time = 0
        self.scatter = None

    def init_ui(self):
        """Initialize layout + plot UI for the PitchPlot widget."""
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(self.colors['background'])
        self._layout.addWidget(self.plot)
        self.add_staff_lines()

        self.plot.getAxis('bottom').setPen(self.colors['label_text'])
        self.plot.getAxis('left').setPen(self.colors['label_text'])
        self.plot.getAxis('bottom').setTextPen(self.colors['label_text'])
        self.plot.getAxis('left').setTextPen(self.colors['label_text'])

    def add_staff_lines(self):
        """Add staff lines to the plot to mimic the treble clef staff."""
        for midi_number in self.staff_lines.values():
            line = pg.InfiniteLine(pos=midi_number, angle=0, pen={'color': self.colors['staff_line'], 'width': 1})
            self.plot.addItem(line)

    def plot_midi(self, MidiData: MidiData):
        """Plot the MIDI data on the pitch plot."""
        self.plot.clear()
        self.add_staff_lines()  # Re-add staff lines after clearing the plot
        self._MidiData = MidiData

        # Get the pitch data from the MIDI file
        pitch_df = self._MidiData.pitch_df

        # Plot the pitch data
        self.bar_height = 1  # Fixed height for all bars
        
        # Calculate the bottom position of each bar so the top aligns with the pitch
        y_positions = pitch_df['pitch']
        
        # Plot all notes as bars
        self.bars = []
        for start, duration, y in zip(pitch_df['start'], pitch_df['duration'], y_positions):
            note_x = start + (duration / 2)
            color = self.colors['MIDI_normal'] if note_x >= self.current_time else self.colors['MIDI_darker']
            bar = pg.BarGraphItem(
                x=note_x,  
                y=y,
                height=self.bar_height, 
                width=duration,
                brush=color,
                name='MIDI')
            self.bars.append(bar)
            self.plot.addItem(bar)

        # The current time line (red)
        self.current_timeline = pg.InfiniteLine(
            pos=self.current_time,
            angle=90,
            pen={'color': self.colors['timeline'], 'width': 2})
        self.plot.addItem(self.current_timeline)

        self.plot.setLabel('bottom', 'Time (s)', color=self.colors['label_text'])  # X-axis label
        self.plot.setLabel('left', 'Pitch (MIDI #)', color=self.colors['label_text'])  # Y-axis label

        # Adjust plot margins
        self.plot.setXRange(self.current_time - self.x_range / 2, self.current_time + self.x_range / 2)
        self.plot.setYRange(60, 80)  # Adjust the Y range to ensure the staff lines are visible


    def plot_user(self, user_pitchdf: pd.DataFrame, onsets=None, min_confidence:float=0.0, 
                  window_size:int=11, note_threshold:float=0.75, harmonic_range:float=0.75):
        """Plot user pitch data on the pitch plot with a confidence filter."""
        # Filter out pitches below the minimum confidence
        user_pitchdf = user_pitchdf[user_pitchdf['confidence'] >= min_confidence]

        if user_pitchdf.empty:
            print('No pitches above the minimum confidence threshold.')
            return
        
        # --- PLOT ONSETS ---
        if hasattr(self, 'onset_lines'):
            for line in self.onset_lines:
                self.plot.removeItem(line)
        else:
            self.onset_lines = []

        if onsets is not None:
            y_center = (user_pitchdf['midi_pitch'].min() + user_pitchdf['midi_pitch'].max()) / 2
            for onset in onsets:
                line = pg.PlotCurveItem(
                    x=[onset, onset], # Small line segment around the onset time
                    y=[self.staff_lines['G4'], self.staff_lines['D5']],
                    pen=pg.mkPen(self.colors['onset'], width=4)
                )
                self.plot.addItem(line)
                self.onset_lines.append(line)

        note_df = PitchAnalyzer.note_segmentation(user_pitchdf, window_size=window_size, threshold=note_threshold)

        # --- PLOT HARMONIC GROUPS ---
        harmonic_groups = PitchAnalyzer.group_harmonics(note_df, harmonic_range=harmonic_range)

        if hasattr(self, 'harmonic_lines'):
            for line in self.harmonic_lines:
                self.plot.removeItem(line)
        else:
            self.harmonic_lines = []

        for group in harmonic_groups:
            
            if len(group) > 1:
                times = [note['time'] for note in group]
                midi_pitches = [note['midi_pitch'] for note in group]
                line = pg.PlotCurveItem(
                    x=times,
                    y=midi_pitches,
                    pen=pg.mkPen(self.colors['onset'], width=15),
                    name='Harmonics'
                )
                self.plot.addItem(line)
                self.harmonic_lines.append(line)

        # --- NOTES LINES ---
        # Clear previous note lines
        if hasattr(self, 'note_lines'):
            for line in self.note_lines:
                self.plot.removeItem(line)
        else:
            self.note_lines = []

        # Add line segments underneath each detected pitch
        if len(note_df) > 1:
            for j in range(len(note_df) - 1):
                # mark start of note with red line
                start_line = pg.PlotCurveItem( 
                    x=[note_df.iloc[j]['time'], note_df.iloc[j]['time']],
                    y=[note_df.iloc[j]['midi_pitch']-1, note_df.iloc[j]['midi_pitch']+1],
                    pen=pg.mkPen(pg.mkColor('r'), width=10),
                    name='Note Start'
                )
                line = pg.PlotCurveItem(
                    x=[note_df.iloc[j]['time'], note_df.iloc[j + 1]['time']],
                    y=[note_df.iloc[j]['midi_pitch'], note_df.iloc[j]['midi_pitch']],
                    pen=pg.mkPen(self.colors['user_note'], width=25),
                    name='Notes'
                )
                self.plot.addItem(line)
                self.note_lines.append(line)
                self.plot.addItem(start_line)
                self.note_lines.append(start_line) 

        # --- PLOT USER PITCHES ---
        # Use the viridis colormap to map confidence values to colors
        colormap = plt.get_cmap('viridis')
        colors = [colormap(confidence) for confidence in user_pitchdf['confidence']]

        # Convert colors to a format that pyqtgraph can use
        colors = [(int(r*255), int(g*255), int(b*255), int(a*255)) for r, g, b, a in colors]
        
        # Remove the previous scatter plot item if it exists
        if hasattr(self, 'scatter'):
            self.plot.removeItem(self.scatter)

        # Create a ScatterPlotItem with the calculated colors
        self.scatter = pg.ScatterPlotItem(
            x=user_pitchdf['time'],
            y=user_pitchdf['midi_pitch'],
            pen=None,
            brush=colors,
            size=5,
            name='Recorded Pitch'
        )

        # Add the scatter plot to the plot widget
        self.plot.addItem(self.scatter)


    def move_plot(self, current_time: float):
        """Move the plot to the current time."""
        self.current_time = current_time
        self.current_timeline.setPos(self.current_time)
        self.plot.setXRange(self.current_time - self.x_range / 2, self.current_time + self.x_range / 2)

        # Update bar colors based on the current time
        for bar in self.bars:
            if bar.opts['x'] < self.current_time:
                bar.setOpts(brush=self.colors['MIDI_darker'])
            else:
                bar.setOpts(brush=self.colors['MIDI_normal'])




