import mido
import pandas as pd

class MidiProcessor:
    def __init__(self):
        pass
    
    @staticmethod
    def create_msg_dict(midi_file):
        """
        Create dictionaries storing all messages and program controls for 
        the MIDI, assuming program is tied to the same channel throughout the piece
        @param:
            - midi_file: str, path to the MIDI file
        @return:
            - message_dict: dict, keys are elapsed time (in sec) and values are 
                            lists of messages all occuring at that time
                -> dict, {elapsed_time: [msg1, msg2, ...]}
            - program_dict: dict, with keys as channel number and values 
                            are a message of type 'program change'
                -> {channel: program_msg}
        """
        midi_data = mido.MidiFile(midi_file)
        message_dict = {}
        program_dict = {}  # Initialize program_dict
        elapsed_time = 0
        for msg in midi_data:
            if not msg.is_meta:
                elapsed_time += msg.time
                if elapsed_time not in message_dict:
                    message_dict[elapsed_time] = []
                message_dict[elapsed_time].append(msg)

                # Check if the message is a 'program_change' message
                if msg.type == 'program_change':
                    program_dict[msg.channel] = msg
        return message_dict, program_dict

    @staticmethod
    def create_pitchdf(message_dict):
        """
        Create a DataFrame from a dictionary of MIDI messages, calculating the duration of each note.
        @param:
            - message_dict: dict, keys are elapsed time (in sec) and values are 
                            lists of messages all occurring at that time
                -> dict, {elapsed_time: [msg1, msg2, ...]}
        @return:
            - pd.DataFrame with columns for start time, pitch, MIDI number, velocity, and duration
        """
        note_start_times = {}  # Dictionary to keep track of note start times
        rows = []  # List to store note details including calculated duration

        for elapsed_time, messages in message_dict.items():
            for msg in messages:
                # Check if the message is a note-related message before accessing the note attribute
                if msg.type in ['note_on', 'note_off']:
                    key = (msg.channel, msg.note)  # Unique key for each note
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Record the start time of the note
                        velocity = msg.velocity
                        note_start_times[key] = (elapsed_time, velocity)
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
        return pitch_df
    

# Example usage

# midi_path = '../resources/midifiles/mozart_vc4_mvt1.mid'
# message_dict, program_dict = create_msg_dict(midi_path)

#pitch_df = create_pitchdf(message_dict)