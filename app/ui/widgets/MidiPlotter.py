from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

class MidiPlotter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pitch_data = {}
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.plotWidget = pg.PlotWidget()
        layout.addWidget(self.plotWidget)
        self.setLayout(layout)

        self.current_time = 0  # Initialize current time

    def plot_midi_data(self, midi_df):
        """Plot the MIDI data on the plot widget"""
        self.plotWidget.clear()
        self.bar_height = 1  # Fixed height for all bars
        # Calculate the bottom position of each bar so the top aligns with the pitch
        y_positions = midi_df['pitch'] - self.bar_height
        self.bars = pg.BarGraphItem(
            x=midi_df['start'],
            y=y_positions,  # Adjusted y position
            height=self.bar_height,  # Fixed height
            width=midi_df['duration'],
            brush='b',
            name='MIDI')
        self.plotWidget.addItem(self.bars)

        # The current time line (red)
        self.currentTimeLine = pg.InfiniteLine(
            pos=self.current_time,
            angle=90,
            pen={'color': 'r', 'width': 2})
        self.plotWidget.addItem(self.currentTimeLine)

        self.plotWidget.setLabel('bottom', 'Time (s)')  # X-axis label
        self.plotWidget.setLabel('left', 'Pitch (MIDI #)')  # Y-axis label

        # Adjust plot margins
        self.plotWidget.setXRange(self.current_time - 5, self.current_time + 5, padding=0.1)

    # def update_plot(self, current_time):
    #     """Updates the plot to reflect the current time based on the slider's position."""
    #     self.current_time = current_time
    #     self.plotWidget.setXRange(self.current_time - 5, self.current_time + 5, padding=0.1)
    #     self.currentTimeLine.setPos(self.current_time)

    def update_plot(self, current_time):
        """Updates the plot to reflect the current time based on the slider's position."""
        self.current_time = current_time
        self.plotWidget.setXRange(self.current_time - 5, self.current_time + 5, padding=0.1)
        self.currentTimeLine.setPos(self.current_time)

        # Update pitch data plot
        if self.pitch_data:
            # Plot pitch data as scatter points
            times = list(self.pitch_data.keys())
            pitches = [self.pitch_data[time] for time in times]
            
            # Normalize pitch data (you might need to adjust this formula based on your specific pitch range)
            normalized_pitches = [(12 * np.log2(pitch / 440) + 69) for pitch in pitches]

            if hasattr(self, 'pitchPlot'):
                self.plotWidget.removeItem(self.pitchPlot)
            self.pitchPlot = pg.PlotCurveItem(
                x=times,
                y=normalized_pitches,
                pen={'color': 'r', 'width': 2},
                name='Recorded Pitch')
            self.plotWidget.addItem(self.pitchPlot)

        else:
            if hasattr(self, 'pitchPlot'):
                self.plotWidget.removeItem(self.pitchPlot)

    def add_pitch_data(self, time, pitch):
        """Add new pitch data and update plot"""
        self.pitch_data[time] = pitch
        self.update_plot(self.current_time)
