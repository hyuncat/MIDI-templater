from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QCheckBox

class InstrumentControl(QWidget):
    channel_toggled = pyqtSignal(int, bool)  # Signal to notify when a channel is toggled

    def __init__(self, instrument_name, channel):
        super().__init__()
        self.channel = channel
        main_layout = QVBoxLayout(self)

        # Create a horizontal layout for the instrument name and checkbox
        row_layout = QHBoxLayout()

        # Instrument name label
        self.label = QLabel(instrument_name)
        row_layout.addWidget(self.label)

        # Channel visibility checkbox without label
        self.checkbox = QCheckBox("")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.toggle_channel)
        row_layout.addWidget(self.checkbox)

        # Add the row layout to the main layout
        main_layout.addLayout(row_layout)

        # Future: Volume control slider
        # self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        # self.volume_slider.setMinimum(0)
        # self.volume_slider.setMaximum(100)
        # self.volume_slider.setValue(50)  # Default volume
        # main_layout.addWidget(self.volume_slider)

        self.setLayout(main_layout)

    def toggle_channel(self, state):
        """Emit signal when channel visibility is toggled"""
        self.channel_toggled.emit(self.channel, state == Qt.CheckState.Checked)