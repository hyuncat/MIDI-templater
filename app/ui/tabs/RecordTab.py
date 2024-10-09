import os
from math import ceil
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QLineEdit

# App config
from app.config import AppConfig
# Audio modules
from app.modules.audio.AudioData import AudioData
from app.modules.audio.AudioRecorder import AudioRecorder
from app.modules.audio.AudioPlayer import AudioPlayer
# MIDI modules
from app.modules.midi.MidiData import MidiData
from app.modules.midi.MidiSynth import MidiSynth
from app.modules.midi.MidiPlayer import MidiPlayer
# Algorithm modules
from app.modules.dtw.PitchDTW import PitchDTW
from archive.PitchAnalyzer import PitchAnalyzer
from app.modules.pitch.pda.PYin import PYin
from app.modules.pitch.models.Onsets import OnsetData
# UI
from app.ui.Slider import Slider
from app.ui.plots.PitchPlot import PitchPlot

class RecordTab(QWidget):
    """Tab for handling initial audio recording/playback"""
    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # Recording/playback variables
        self.is_recording = False
        self.is_midi_playing = False
        self.is_user_playing = False

        # Initialize the MIDI/user audio data
        # (Can be omitted later to allow custom MIDI uploads and
        # user audio recording within the app itself)
        # ---
        self.MIDI_FILE = "fugue.mid" # resources/midi/...
        self.SOUNDFONT_FILE = "MuseScore_General.sf3" # resources/...
        self.USER_AUDIO_FILE = "user_fugue2.mp3" # resources/audio/...

        self.init_midi(self.MIDI_FILE, self.SOUNDFONT_FILE)
        self.init_user_audio(self.USER_AUDIO_FILE)
        self.init_pitch_plot()
        self.init_slider()
        self.init_playback_buttons()
    
    def init_midi(self, midi_file: str, soundfont_file: str):
        """
        Initializes MIDI data, synth, and player for use in the app.
        Expects the file name as input and parses filepath as the app/resources 
        folder automatically.

        Args:
            midi_file (str): MIDI file to load
            soundfont_file (str): Soundfont file to load
        """
        print(f"Starting app with MIDI file {midi_file}...")

        # Get MIDI/soundfont file paths
        app_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        soundfont_filepath = os.path.join(app_directory, 'resources', soundfont_file)
        midi_filepath = os.path.join(app_directory, 'resources', 'midi', midi_file)

        # Initialize MIDI data, synth, and player
        self._MidiData = MidiData(midi_filepath)
        self._MidiSynth = MidiSynth(soundfont_filepath)
        self._MidiPlayer = MidiPlayer(self._MidiSynth)
        self._MidiPlayer.load_midi(self._MidiData) # Load MIDI data into the player

    def init_user_audio(self, user_audio_file: str) -> None:
        """
        Preload the app with an audio recording to perform analysis.
        """
        # Get audio file path
        app_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        audio_filepath = os.path.join(app_directory, 'resources', 'audio', user_audio_file)

        # Initialize audio data, recorder, and player
        # self._AudioRecorder = AudioRecorder()
        
        self._AudioData = AudioData()
        self._AudioData.load_data(audio_filepath)

        self._AudioPlayer = AudioPlayer()
        self._AudioPlayer.load_audio_data(self._AudioData)
        print(f"Preloaded user audio: {user_audio_file}")

        self.pitches, self.best_prob_pitches = PYin.pyin(self._AudioData.data, mean_threshold=0.3)
        self.onset_data = OnsetData(self._AudioData)
        # self.onset_times = self.os.detect_onsets(self._AudioData)
        self.onset_data.detect_pitch_changes(self.best_prob_pitches, window_size=30, threshold=0.6)
        self.onset_data.combine_onsets(combine_threshold=0.05)
    
    def init_pitch_plot(self):
        self._PitchPlot = PitchPlot()
        self._PitchPlot.plot_midi(self._MidiData)

        # Plot these if user preloads in data
        self._PitchPlot.plot_onsets(self.onset_data.onset_df)
        # self._PitchPlot.plot_notes(self.note_df)
        self._PitchPlot.plot_pitches(self.best_prob_pitches)
        self._layout.addWidget(self._PitchPlot)

    def init_slider(self):
        self._Slider = Slider(self._MidiData) # Init slider with current MIDI data
        self._Slider.slider_changed.connect(self.handle_slider_change)

        # Update the slider max value if the audio file is longer than the MIDI file
        if hasattr(self, "_AudioData"):
            audio_length = ceil(self._AudioData.get_length())
            if audio_length > self._MidiData.get_length():
                slider_ticks = audio_length * self._Slider.TICKS_PER_SEC
                self._Slider.update_slider_max(slider_ticks)
        
        self._layout.addWidget(self._Slider)
    
    def init_playback_buttons(self):
        """Init playback/record buttons"""
        # Create a horizontal layout for the playback and recording buttons
        self.buttonLayout = QHBoxLayout()

        # Add togglePlay and toggleRecord buttons
        self.midi_play_button = QPushButton('Play MIDI')
        self.midi_play_button.clicked.connect(self.toggle_midi)
        self.buttonLayout.addWidget(self.midi_play_button)

        # self.record_button = QPushButton('Start recording')
        # self.record_button.clicked.connect(self.toggle_record)
        # self.buttonLayout.addWidget(self.record_button)

        # Listen back to audio
        self.user_play_button = QPushButton('Play Recorded Audio')
        self.user_play_button.clicked.connect(self.toggle_user_playback)
        self.buttonLayout.addWidget(self.user_play_button)

        # DTW button
        self.dtw_button = QPushButton('Align to MIDI')
        self.dtw_button.clicked.connect(self.dtw_align)
        self.buttonLayout.addWidget(self.dtw_button)

        self._layout.addLayout(self.buttonLayout)

    def toggle_midi(self):
        """Toggle the MIDI playback on and off."""
        #TODO: Make slider not stop if at least one playback is running
        if self.is_midi_playing: # Pause timer if playing
            self._Slider.stop_timer()
            self.is_midi_playing = False
            self.midi_play_button.setText('Play MIDI')
            self._MidiPlayer.pause()
        else: # Unpause timer if not playing
            self._Slider.start_timer()
            self.is_midi_playing = True
            self.midi_play_button.setText('Pause MIDI')
            start_time = self._Slider.get_current_time()
            self._MidiPlayer.play(start_time=start_time)

    def toggle_user_playback(self):
        """Toggle user's recorded audio playback on/off"""
        if not hasattr(self, "_AudioPlayer"):
            return
        
        if self.is_user_playing:
            self._Slider.stop_timer()
            self.user_play_button.setText('Play Recorded Audio')
            self._AudioPlayer.pause()
            self.is_user_playing = False
        else:
            self._Slider.start_timer()
            self.user_play_button.setText('Pause Recorded Audio')
            start_time = self._Slider.get_current_time()
            self._AudioPlayer.play(start_time=start_time)
            self.is_user_playing = True

    def handle_slider_change(self, value):
        """Handle the slider change event, e.g., seeking in a MIDI playback"""
        #TODO: Handle seeking during audio/midi playback
        current_time = self._Slider.get_current_time() 
        self._PitchPlot.move_plot(current_time)

        if value < self._Slider.slider.minimum():
            self._Slider.slider.setValue(self._Slider.slider.minimum())

        if value >= self._Slider.slider.maximum():
            self._Slider.stop_timer()
            # Pause MIDI playback if it's currently playing
            self.is_midi_playing = False
            self.midi_play_button.setText('Play MIDI')
            
            if hasattr(self, "_MidiPlayer"):
                self._MidiPlayer.pause()

            # Pause user audio playback if it's currently playing
            self.is_user_playing = False
            self.user_play_button.setText('Play Recorded Audio')
            if hasattr(self, "_AudioPlayer"):
                self._AudioPlayer.pause()

    def dtw_align(self):
        print("Align notes! (Implemented later)")