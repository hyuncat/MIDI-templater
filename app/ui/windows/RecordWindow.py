# app/ui/widgets/window_one.py
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from music21 import converter, instrument
from ..widgets.InstrumentControl import InstrumentControl

class RecordWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.label = QLabel("Record violin audio here...", self)
        layout.addWidget(self.label)

        # Determine the correct path to the MIDI file
        current_dir = os.path.dirname(__file__)  # Gets the directory of the current file
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # Navigate up to the project root
        midi_path = os.path.join(project_root, 'resources', 'midifiles', 'mozart_vc4_mvt1.mid')

        # Load MIDI with music21 and parse instruments
        # midi_data = converter.parse(midi_path)
        # self.display_instruments(midi_data, layout)

    def status_message(self):
        return "Recording..."

    # def display_instruments(self, midi_data, layout):
    #     # Iterate over all parts and check for instruments
    #     for part in midi_data.parts:
    #         inst = instrument.partitionByInstrument(part)
    #         if inst:  # Check if the part has instruments
    #             for i in inst:
    #                 name = i.getInstrument().instrumentName if i.getInstrument() else "Unknown Instrument"
    #                 inst_widget = InstrumentControl(name)
    #                 layout.addWidget(inst_widget)