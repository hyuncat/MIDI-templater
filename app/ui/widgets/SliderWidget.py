from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QSlider, QVBoxLayout

class SliderWidget(QWidget):
    sliderChanged = pyqtSignal(int)  # Signal to emit when the slider value changes
    sliderMoved = pyqtSignal(int)    # Signal to emit when the slider is moved

    def __init__(self, midiPlayer, audioRecorderThread, audioPlaybackThread):
        super().__init__()
        self.midiPlayer = midiPlayer
        self.audioRecorderThread = audioRecorderThread
        self.audioPlaybackThread = audioPlaybackThread
        self.timer_speed = 100 # update every 100 ms
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.slider = QSlider(Qt.Orientation.Horizontal)  # The QSlider object
        self.layout.addWidget(self.slider)

        # Set slider range + initial value
        self.slider.setValue(0)
        self.slider.setMinimum(0)

        # Connect slider signals
        self.slider.sliderMoved.connect(self.emit_slider_moved)
        self.slider.valueChanged.connect(self.emit_slider_change)

        # Timer to update slider position
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_slider_position)
        self.current_slider = 0 # in 0.1 sec, raw slider value
        self.current_time = 0 # in sec, converted from slider value

        self.midi_length_seconds = int(self.midiPlayer.get_end_time())
        self.slider.setMaximum(self.midi_length_seconds * 10)  # Adjust the range

    def emit_slider_change(self, value):
        self.sliderChanged.emit(value)  # Emit the signal with the current slider value

    def emit_slider_moved(self, value):
        self.current_slider = value
        self.current_time = value / 10
        self.sliderMoved.emit(value)    # Emit the signal with the current slider value

    def start_timer(self):
        self.timer.start(self.timer_speed)  # Update every 100 ms

    def stop_timer(self):
        self.timer.stop()
    
    def change_timer_speed(self, change_factor):
        self.timer_speed = int(self.timer_speed / change_factor)
        self.timer.setInterval(self.timer_speed)

    def get_current_time(self):
        return self.current_time

    def update_slider_position(self):
        """Update the slider position based on the current playback time"""
        is_midi_playing = (hasattr(self, 'midiPlayer') and self.midiPlayer.get_is_playing())
        is_recording = (hasattr(self, 'audioRecorderThread') and self.audioRecorderThread.get_is_recording())
        is_rec_playing = (hasattr(self, 'audioPlaybackThread') and self.audioPlaybackThread.get_is_playing())

        if is_midi_playing or is_recording or is_rec_playing:
            self.current_slider += 1
            self.current_time = self.current_slider / 10

            if is_midi_playing:
                #TODO: make slider update smoother by using the next message time to inform move amt
                midi_time = self.midiPlayer.get_current_time()
                if abs(self.current_time - midi_time) > 0.2:
                    self.current_slider = midi_time * 10
                    self.current_time = midi_time
            # elif is_rec_playing:
            #     audio_time = self.audioPlaybackThread.get_playback_time()
            #     if abs(self.current_time - audio_time) > 0.2:
            #         self.current_slider = audio_time * 10
            #         self.current_time = audio_time
            
            # current_time = self.midiPlayer.get_current_time()  # Assume this method exists
            self.slider.setValue(int(self.current_slider))
    

    def set_midi_player(self, midiPlayer):
        """Set the MIDI player to get the current playback time"""
        self.midiPlayer = midiPlayer

    def setMaximum(self, max_value):
        self.slider.setMaximum(max_value)

    def setValue(self, value):
        self.slider.setValue(value)
