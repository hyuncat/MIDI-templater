# violin-CV

**Current goal:** Real-time pitch correction

## MidiPlotter.py

GUI class in `MidiPlotter.py` (Can you guess the piece?)

<img width="400" alt="Screenshot 2024-06-19 at 3 23 32 AM" src="https://github.com/hyuncat/violin-CV/assets/114366569/9ff7ffa6-a111-4ffe-8cc2-725f2f5f8245">

#### Plot axes and values:
- x-axis: Seconds since beginning of recording
- y-axis: Midi number (corresponding to a pitch)
- Blue piano rolls: MIDI violin pitches
- Red line: User audio input

### Currently

Currently, the program accepts a MIDI input and produces a piano roll visualization for the 'Violin' instrument. When users press **play**, the MIDI starts playing and it also begins to capture the audio, where the user can play violin. Their recorded audio pitch is then plotted on the graph and can be compared in real time to what the MIDI should sound like.

To extract the pitch from the audio frequency data, the program uses the PitchYin algorithm from Essentia library (returning Hz + condifence). PyQT6 is used for the GUI and pyqtgraph offers fast real-time plotting updates.


## Todo

Next goal is to integrate two modes of pitch correction:

### 1. Out of tune

Given a tuning standard (e.g., A=440Hz), provide users a notification within some threshold that they are out of tune, and by how much.

### 2. Wrong note

Given a YouTube link, convert to MIDI or MP3 and store as an array of pitches to compare user input to, and correct users if they play a wrong note - what they did play, and what they should have played.
