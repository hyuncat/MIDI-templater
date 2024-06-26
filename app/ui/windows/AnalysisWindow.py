import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QLineEdit, QHBoxLayout
from ..widgets.SliderWidget import SliderWidget
from app.core.MidiPlayer import MidiPlayer
from app.core.MidiSynthesizer import MidiSynthesizer
from app.core.MidiProcessor import MidiProcessor
from app.core.AudioRecorder import SharedAudioData, AudioPlaybackThread, AudioRecorderThread
from ..widgets.MidiPlotter import MidiPlotter
from ..widgets.InstrumentControl import InstrumentControl

class AnalysisWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupMidi()
        self.initUI()

        self.is_playing = False
        self.is_recording = False
        self.is_user_playing = False

        self.pitch_data = {}
        

    def setupMidi(self):
        # Determine the correct path to the MIDI file
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        soundfont_path = os.path.join(project_root, 'resources', 'soundfonts', 'MuseScore_General.sf3')
        midi_path = os.path.join(project_root, 'resources', 'midifiles', 'mozart_vc4_mvt1.mid')

        self.midiSynthesizer = MidiSynthesizer(soundfont_path)
        self.midiPlayer = MidiPlayer(self.midiSynthesizer)
        self.midiPlayer.load_midi(midi_path)

        self.shared_data = SharedAudioData(self.midiPlayer.get_end_time())

    def initUI(self):
        layout = QVBoxLayout(self)
        self.label = QLabel("Analyze and playback recorded audio here...", self)
        layout.addWidget(self.label)

        self.setupMidi()  # Initialize the MIDI player and synthesizer

        # Initialize the MIDI plot widget
        self.midiPlotter = MidiPlotter()
        self.midiPlotter.plot_midi_data(self.midiPlayer.midi_df)  # Plot initial MIDI data
        layout.addWidget(self.midiPlotter)

        
        # Initialize the audio recorder and playback
        # self.audioRecorder = AudioRecorder(self.midiPlayer.get_end_time())
        # self.audioRecorder.recording_started.connect(self.on_recording_started)
        # self.audioRecorder.recording_stopped.connect(self.on_recording_stopped)
        # self.audioRecorder.playback_info.connect(self.display_playback_info)
        
        # Initialize the audio recorder and playback threads
        self.audioRecorderThread = AudioRecorderThread(self.shared_data)
        self.audioPlaybackThread = AudioPlaybackThread(self.shared_data)

        self.audioRecorderThread.recording_started.connect(self.on_recording_started)
        self.audioRecorderThread.recording_stopped.connect(self.on_recording_stopped)
        self.audioRecorderThread.pitch_data_updated.connect(self.update_pitch_plot)
    
        # Initialize the slider widget with given MIDI player
        self.sliderWidget = SliderWidget(self.midiPlayer, self.audioRecorderThread, self.audioPlaybackThread)
        self.sliderWidget.sliderChanged.connect(self.handle_slider_change)
        layout.addWidget(self.sliderWidget)

       # Create a horizontal layout for the playback and recording buttons
        self.buttonLayout = QHBoxLayout()

        # Add togglePlay and toggleRecord buttons
        self.togglePlayButton = QPushButton('Play MIDI')
        self.togglePlayButton.clicked.connect(self.toggle_midi)
        self.buttonLayout.addWidget(self.togglePlayButton)

        self.toggleRecordButton = QPushButton('Start recording')
        self.toggleRecordButton.clicked.connect(self.toggle_record)
        self.buttonLayout.addWidget(self.toggleRecordButton)

        # Listen back to audio
        self.toggleUserPlaybackButton = QPushButton('Play Recorded Audio')
        self.toggleUserPlaybackButton.clicked.connect(self.toggle_user_playback)
        self.buttonLayout.addWidget(self.toggleUserPlaybackButton)

        layout.addLayout(self.buttonLayout)

        # Add tempo control input and button on the same row
        self.tempoControlLayout = QHBoxLayout()
        self.tempoInput = QLineEdit()
        self.tempoInput.setPlaceholderText("Enter tempo factor (e.g., 1.5)")
        self.tempoControlLayout.addWidget(self.tempoInput)
        self.changeTempoButton = QPushButton("Change Tempo")
        self.changeTempoButton.clicked.connect(self.change_tempo)
        self.tempoControlLayout.addWidget(self.changeTempoButton)
        layout.addLayout(self.tempoControlLayout)

        # Initialize a horizontal layout to place instrument controls on the same row
        self.instrumentRowLayout = QHBoxLayout()

        for channel in self.midiPlayer.get_channels():
            instrument_name = f"Instrument {channel}"  # Replace with actual instrument names if available
            instrument_control = InstrumentControl(instrument_name, channel)
            instrument_control.channel_toggled.connect(self.toggle_channel)
            self.instrumentRowLayout.addWidget(instrument_control)

        # Add the row layout to the main layout
        layout.addLayout(self.instrumentRowLayout)

        self.recordingInfo = QTextEdit()
        self.recordingInfo.setReadOnly(True)
        layout.addWidget(self.recordingInfo)


    def toggle_midi(self):
        """Toggle the MIDI playback on and off."""
        if self.is_playing: # Pause timer if playing
            self.sliderWidget.stop_timer()
            self.is_playing = False
            self.togglePlayButton.setText('Play MIDI')
        else: # Unpause timer if not playing
            self.sliderWidget.start_timer()
            self.is_playing = True
            self.togglePlayButton.setText('Pause MIDI')
        self.midiPlayer.pause_midi(self.sliderWidget.current_time)

    def toggle_record(self):
        if self.is_recording:
            self.sliderWidget.stop_timer()
            self.audioRecorderThread.stop_recording(self.sliderWidget.current_time)
            self.audioRecorderThread.stop()
            self.toggleRecordButton.setText('Start Recording')
            self.is_recording = False
        else:
            self.sliderWidget.start_timer()
            self.audioRecorderThread.start()
            self.audioRecorderThread.start_recording(self.sliderWidget.current_time)
            self.toggleRecordButton.setText('Stop Recording')
            self.is_recording = True

    def toggle_user_playback(self):
        if self.is_user_playing:
            self.sliderWidget.stop_timer()
            self.is_user_playing = False
            self.toggleUserPlaybackButton.setText('Play Recorded Audio')
            self.audioPlaybackThread.pause_audio()
        else:
            self.sliderWidget.start_timer()
            self.is_user_playing = True
            self.toggleUserPlaybackButton.setText('Pause Recorded Audio')
            start_from = self.sliderWidget.get_current_time() / 10.0
            self.audioPlaybackThread.play_audio(start_from=start_from)

    def on_recording_started(self, start_time):
        self.recordingInfo.append(f"Recording started at: {start_time}")

    def on_recording_stopped(self, end_time):
        self.recordingInfo.append(f"Recording stopped at: {end_time}")

    def display_playback_info(self, info_text):
        self.recordingInfo.append(info_text)

    def toggle_channel(self, channel, state):
        """Toggle the visibility of a channel."""
        if state:
            if channel not in self.midiPlayer.playing_channels:
                self.midiPlayer.playing_channels.append(channel)
        else:
            if channel in self.midiPlayer.playing_channels:
                self.midiPlayer.playing_channels.remove(channel)
        self.midiPlayer.change_channels(self.midiPlayer.playing_channels)
        # self.update_midi_plot()


    def status_message(self):
        return "Analyzing..."

    def change_tempo(self):
        """Change the tempo of the MIDI playback."""
        try:
            tempo_factor = float(self.tempoInput.text())
            self.midiPlayer.change_tempo(tempo_factor)
            self.sliderWidget.change_timer_speed(tempo_factor)
            self.recordingInfo.append(f"Tempo changed to: {tempo_factor}")
            self.sliderWidget.setMaximum(int(self.midiPlayer.get_end_time() * 10))
        except ValueError:
            self.recordingInfo.append("Invalid tempo factor. Please enter a numeric value.")

    

    def handle_slider_change(self, value):
        """Handle the slider change event, e.g., seeking in a MIDI playback"""
        print(f"Slider value changed to: {value}")
        self.midiPlayer.seek(value / 10.0)  # Seek to the appropriate time
        self.midiPlotter.update_plot(value / 10.0)

    def update_pitch_plot(self, time, pitch):
        """Update the plot with new pitch data"""
        self.pitch_data[time] = pitch
        self.midiPlotter.add_pitch_data(time, pitch)