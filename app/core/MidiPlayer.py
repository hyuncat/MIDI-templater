import time
import threading
from .MidiProcessor import MidiProcessor
from .MidiSynthesizer import MidiSynthesizer

class MidiPlayer:
    """Final class with easy functions to call in the app"""
    def __init__(self, synthesizer: MidiSynthesizer):
        self.synthesizer = synthesizer
        self.play_thread = None
        self.stop_event = threading.Event()
        self.current_time = 0
        self.is_playing = False
        self.all_channels = []
        self.playing_channels = []

    def load_midi(self, midi_file):
        """Creates message and program dictionaries containing all messages 
        and program changes from the MIDI file."""
        self.message_dict, self.program_dict = MidiProcessor.create_msg_dict(midi_file)
        self.midi_df = MidiProcessor.create_pitchdf(self.message_dict)

        # Set channels
        self.all_channels = list(self.program_dict.keys())
        self.playing_channels = self.all_channels

    def get_channels(self):
        """Returns the list of channels in the MIDI file."""
        return list(self.program_dict.keys())
    
    def get_current_time(self):
        """Returns the current time of the MIDI playback."""
        return self.current_time
    
    def get_end_time(self):
        """Get the last key in the message dictionary to get the end time."""
        if self.midi_df is None:
            return 0
        return self.midi_df['start'].max() + self.midi_df['duration'].max()

    
    def get_is_playing(self):
        """Returns True if the MIDI player is currently playing."""
        return self.is_playing
    
    def change_channels(self, channels):
        """Change the channels to play from the MIDI file."""
        self.playing_channels = channels

    def play_midi(self, start_from=0, channels=None):
        """
        Plays a MIDI file using the synthesizer with the given instruments from a particular time.

        @param:
            - start_from (int): Time in seconds to start playing from
            - channels (list): List of channels to play
        """
        if self.play_thread is not None and self.play_thread.is_alive():
            self.stop_event.set()
            self.synthesizer.stop_all_notes()
            self.play_thread.join()

        self.stop_event.clear()
        self.play_thread = threading.Thread(target=self._play_midi, args=(start_from, channels))
        self.play_thread.start()

    def _play_midi(self, start_from, channels):
        if self.message_dict is None or self.program_dict is None:
            print("No MIDI file loaded. Exiting.")
            return

        self.is_playing = True
        self.synthesizer.unpause()

        # Set all channels with a respective program before starting
        for channel, program_msg in self.program_dict.items():
            self.synthesizer.program_select(program_msg.channel, 0, program_msg.program)
            print(f'Set program {program_msg.program} on channel {program_msg.channel}')
        
        # Assuming messages are sorted by time, no need to sort again
        times = list(self.message_dict.keys())
        
        # Find the index to start from based on the start_from time
        start_index = next((i for i, t in enumerate(times) if t >= start_from), None)
        if start_index is None:
            print("Start time is beyond the last message. Exiting.")
            return
        
        for i in range(start_index, len(times)):
            if self.stop_event.is_set():
                break
            start_time = times[i]
            self.current_time = start_time
            messages = self.message_dict[start_time]
            for msg in messages:
                # Only play messages from specified channels ('instruments')
                if msg.channel in self.playing_channels: 
                    self.synthesizer.handle_midi_message(msg)
            if i + 1 < len(times):
                next_start_time = times[i + 1]
                sleep_duration = next_start_time - start_time
                time.sleep(sleep_duration)


    def pause_midi(self, pause_time=None):
        """Pauses the MIDI playback."""
        if self.play_thread is not None and self.play_thread.is_alive():
            self.stop_event.set()
            self.play_thread.join()

        if pause_time is not None:
            self.current_time = pause_time
        
        # Pause if playing
        if self.is_playing is True:
            self.synthesizer.pause()
            self.is_playing = False
        # Unpause if not
        elif self.is_playing is False:
            self.synthesizer.unpause()
            self.is_playing = True
            self.play_midi(start_from=self.current_time, channels=self.get_channels())

    def get_current_time(self):
        """Returns the current playback time in seconds."""
        return self.current_time

    def seek(self, seek_to_time):
        """Seeks to a specific time in the MIDI file."""
        # self.pause_midi()
        self.current_time = seek_to_time

    def change_tempo(self, change_factor):
        """
        Warp tempo by multiplying all times by a factor.
        Ex: change_factor = 2 will half the tempo, change_factor = 0.5 will double it tempo.
        """
        if change_factor <= 0:
            print("Invalid tempo change factor. Exiting.")
            return
        elif self.message_dict is None:
            print("No MIDI file loaded. Exiting.")
            return
        self.message_dict = {k / change_factor: v for k, v in self.message_dict.items()}
        self.midi_df = MidiProcessor.create_pitchdf(self.message_dict)