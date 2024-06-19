# violin-CV

**Current goal:** Real-time pitch correction

## MidiPlotter.py

GUI class in `MidiPlotter.py` (Can you guess the piece?)

<img width="420" alt="Screenshot 2024-06-19 at 5 01 06 AM" src="https://github.com/hyuncat/violin-CV/assets/114366569/3a6ba656-5e65-4535-ad6d-c6f16e76b38d">

#### Plot axes and values:
- x-axis: Seconds since beginning of recording
- y-axis: Midi number (corresponding to a pitch)
- Blue piano rolls: MIDI violin pitches
- Red line: User audio input

### Currently

Currently, the program accepts a MIDI input and produces a piano roll visualization for the 'Violin' instrument. When users press **play**, the MIDI starts playing and it also begins to capture the audio, where the user can play violin. Their recorded audio pitch is then plotted on the graph and can be compared in real time to what the MIDI should sound like.

To extract the pitch from the audio frequency data, the program uses the PitchYin algorithm from Essentia library (returning Hz + condifence). PyQT6 is used for the GUI and pyqtgraph offers fast real-time plotting updates.


## Todo

Next goal is to create a minimum viable use case for the program. This includes integrating the following features:

### 1. Able to playback recorded audio from earlier points

Currently, the program allows you to scrub to early parts and see how your performance differs from the MIDI, but there is no way to play back the recording it captured. The aim is to allow playback of either: 

- Just the MIDI
- Just their recorded audio
- Or both overlayed on top of each other.

To provide audio feedback / review of the difference in their playing alongside the visual differences.

### 2. MIDI tempo control

While currently the program only plays back and displays the MIDI data at a set tempo, our second goal is to dynamically scale all the note lengths longer or shorter depending on the user's desired tempo. Using the scaled MIDI data, we want to generate and play back the desired slower/faster audio waveform in the program.

(And perhaps in the future, we are considering to implement dynamic time warping to 'snap' a user's pitches to the MIDI, so even without being perfectly in time they can isolate their performance analysis to just their intonation.)
