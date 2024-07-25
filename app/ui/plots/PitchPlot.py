from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from app.modules.midi.MidiData import MidiData

class PitchPlot(QWidget):
    def __init__(self):
        super().__init__()

        self.colors = {
            'background': '#ECECEC',
            'label_text': '#000000',  # Black color
            'MIDI_normal': '#000000',  # Black
            'MIDI_darker': '#377333',  # Green
            'user': '#0000FF',  # Blue
            'timeline': '#FF0000',  # Red color
            'staff_line': '#A9A9A9'  # Dark Gray for staff lines
        }

        self.staff_lines = {
            'E4': 64,
            'G4': 67,
            'B4': 71,
            'D5': 74,
            'F5': 77
        }

        self.init_ui()
        self.x_range = 10  # Show 10 seconds of pitch data at once
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

    def plot_user(self, pitch_values, pitch_confidences, pitch_times, min_confidence=0.0):
        """Plot user pitch data on the pitch plot with a confidence filter."""
        # Filter out pitches below the minimum confidence
        filtered_data = [
            (pitch, confidence, time) for pitch, confidence, time in zip(pitch_values, pitch_confidences, pitch_times)
            if confidence >= min_confidence
        ]

        if not filtered_data:
            return

        filtered_pitches, filtered_confidences, filtered_times = zip(*filtered_data)

        # Normalize pitch data to MIDI numbers
        normalized_pitches = [(12 * np.log(pitch / 220) / np.log(2) + 57) for pitch in filtered_pitches]

        # Use the viridis colormap to map confidence values to colors
        colormap = plt.get_cmap('viridis')
        colors = [colormap(confidence) for confidence in filtered_confidences]

        # Convert colors to a format that pyqtgraph can use
        colors = [(int(r*255), int(g*255), int(b*255), int(a*255)) for r, g, b, a in colors]
        
        # Remove the previous scatter plot item if it exists
        if self.scatter is not None:
            self.plot.removeItem(self.scatter)

        # Create a ScatterPlotItem with the calculated colors
        self.scatter = pg.ScatterPlotItem(
            x=filtered_times,
            y=normalized_pitches,
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
