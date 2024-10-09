import mido
import pandas as pd
import logging

class MidiLoader:
    def __init__(self):
        """
        MidiLoader is a class for parsing MIDI files into playable & readable dfs.
        
        Creates the following
        - message_dict: dict, {elapsed_time: [msg1, msg2, ...]}
        - program_dict: dict, {cannel: program_change (Message)}
        - pitch_df: DataFrame with columns for 
            -> start_time, channel, pitch, velocity, duration
        """
        pass

    @staticmethod
    def parse_midi(midi_file_path: str) -> tuple[dict, dict, pd.DataFrame]:
        """
        Parse a MIDI file into a message_dict, program_dict, and pitch_df.
        """
        midi_data = mido.MidiFile(midi_file_path) # Creates array of Message objects

        message_dict = {}
        program_dict = {}

        elapsed_time = 0
        for msg in midi_data:
            if msg.is_meta:
                continue # Skip meta messages like tempo change, key signature, etc.
            
            time_since_last_msg = msg.time
            elapsed_time += time_since_last_msg

            if elapsed_time not in message_dict:
                message_dict[elapsed_time] = []

            message_dict[elapsed_time].append(msg)

            # Add 'program_change' messages to program_dict
            if msg.type == 'program_change':
                program_dict[msg.channel] = msg
        
        pitch_df = MidiLoader.create_pitchdf(message_dict)
        return message_dict, program_dict, pitch_df


    @staticmethod
    def create_pitchdf(message_dict: dict) -> pd.DataFrame:
        """
        Internal function to create a more interpretable DataFrame of pitches, 
        velocity, and duration from a message_dict object
        @param:
            - message_dict: dict, keys are elapsed time (in sec) and values are 
                            lists of messages all occurring at that time
                -> dict, {elapsed_time: [msg1, msg2, ...]}
        @return:
            - pitch_df: Dataframe with columns for 
                -> start time, pitch, MIDI number, velocity, duration
        """
        note_start_times = {}  # Dictionary to keep track of note start times
        rows = []  # List to store note details including calculated duration

        for elapsed_time, messages in message_dict.items():
            for msg in messages: # Iterate through all messages at a given time

                # Check if the message is a note-related message before accessing the note attribute
                if msg.type in ['note_on', 'note_off']:
                    key = (msg.channel, msg.note)  # Unique key for each note

                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Record the start time of the note
                        velocity = msg.velocity
                        note_start_times[key] = (elapsed_time, velocity)

                    # Stop note and compute duration
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        # Calculate duration and prepare the row for DataFrame
                        if key in note_start_times:
                            start_time, velocity = note_start_times[key]
                            duration = elapsed_time - start_time
                            row = [start_time, msg.channel, msg.note, velocity, duration]
                            rows.append(row)
                            del note_start_times[key]  # Remove the note from start times

        # Create DataFrame from the rows
        pitch_df = pd.DataFrame(rows, columns=['start', 'channel', 'pitch', 'velocity', 'duration'])
        
        # Create frequency col (https://www.music.mcgill.ca/~gary/307/week1/node28.html)
        pitch_df['frequency'] = 440 * (2 ** ((pitch_df['pitch'] - 69) / 12))
        return pitch_df


class MidiData:
    """
    MidiData stores all data and methods related to a single midifile.
    
    Stores the following data structures:
        - message_dict: dict, {elapsed_time: [msg1, msg2, ...]}
        - program_dict: dict, {cannel: program_change (Message)}
        - pitch_df: DataFrame with columns for 
            -> start_time, channel, pitch, velocity, duration
    
    And provides the following methods:
        - get_length() -> float: returns the length of the MIDI file in seconds
        - get_channels() -> list: returns the list of channels in the MIDI file
    """

    def __init__(self, midi_file_path: str):
        """
        Initialize the MidiData object by parsing the given MIDI file.
        """
        self.message_dict, self.program_dict, self.pitch_df = MidiLoader.parse_midi(midi_file_path)

    def get_length(self) -> float:
        """Get the length of the MIDI file in seconds."""
        if self.pitch_df is None:
            logging.error("No MIDI data loaded!")
            return 0
        
        return self.pitch_df['start'].max() + self.pitch_df['duration'].max()
    
    def get_channels(self) -> list:
        """Get the list of channels in the MIDI file."""
        return list(self.program_dict.keys())