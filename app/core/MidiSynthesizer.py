import mido
import rtmidi
from mido import Message
import fluidsynth
import time

class MidiSynthesizer:
    def __init__(self, soundfont_path):
        """Class to feed midi messages to an audio channel with fluidsynth."""
        self.synth = fluidsynth.Synth()
        # Start the synthesizer with the appropriate audio driver
        self.synth.start(driver="coreaudio")  # Change to 'alsa', 'dsound', etc., based on your OS
        self.sf_id = self.load_soundfont(soundfont_path)

        # Track playing notes and pause status
        self.playing_notes = {}  # Track playing notes: {channel: [midi_number, ...]}
        self.paused = False
    
    def load_soundfont(self, soundfont_path):
        """
        Load a soundfont from a filepath, e.g. 'MuseScore_General.sf3'
        @return: The soundfont ID.
        """
        sf_id = self.synth.sfload(soundfont_path)
        if sf_id == -1:
            raise ValueError("Soundfont failed to load")
        return sf_id

    def program_select(self, channel, bank, preset):
        """
        Select the instrument program for a specific channel.

        @param:
            channel (int): MIDI channel (0-15)
            bank (int): SoundFont bank number
            preset (int): Preset number (program number) within the bank
        """
        self.synth.program_select(channel, self.sf_id, bank, preset)

    def play_note(self, channel, midi_number, velocity):
        """
        Play a note on a specific channel with given parameters.

        @param:
            - channel (int): MIDI channel on which to play the note
            - midi_number (int): MIDI number of the note to play
            - velocity (int): Velocity of the note (controls volume and articulation)
        """
        if not self.paused:
            self.synth.noteon(channel, midi_number, velocity)
            # Also add notes to currently playing notes
            if channel not in self.playing_notes:
                self.playing_notes[channel] = []
            self.playing_notes[channel].append(midi_number)

    def stop_note(self, channel, midi_number):
        """
        Stop a note on a specific channel.

        @param:
            - channel (int): MIDI channel on which to stop the note
            - midi_number (int): MIDI number of the note to stop
        """
        self.synth.noteoff(channel, midi_number)
        if channel in self.playing_notes:
            self.playing_notes[channel] = [note for note in self.playing_notes[channel] if note != midi_number]

    def pause(self):
        """Stop all currently playing notes"""
        if not self.paused:
            for channel, notes in self.playing_notes.items():
                for midi_number in notes:
                    self.synth.noteoff(channel, midi_number)
            self.paused = True

    def unpause(self):
        """Resume playing notes"""
        self.paused = False

    def stop_all_notes(self):
        """Stop all currently playing notes on all channels."""
        for channel in range(16):
            for note in range(128):
                self.synth.noteoff(channel, note)
    
    def stop(self):
        """
        Completely stop the synthesizer and clean up resources.
        """
        self.synth.delete()

    def handle_midi_message(self, msg: Message):
        """Handle different types of MIDI messages."""
        if self.paused:
            return
        if msg.type == 'note_on':
            self.play_note(msg.channel, msg.note, msg.velocity)
        elif msg.type == 'note_off':
            self.stop_note(msg.channel, msg.note)
        elif msg.type == 'program_change':
            self.program_select(msg.channel, 0, msg.program)

