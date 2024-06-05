# violin-CV

**Current goal:** Real-time pitch correction

## PitchAnalyzer.py

<img width="300" alt="Screenshot 2024-06-05 at 10 23 55 AM" src="https://github.com/hyuncat/violin-CV/assets/114366569/d9b60bd1-9915-4641-b756-2de21356d23a">

Currently, the program uses the Essentia library to analyze pitch from audio input (returning Hz + condifence), and PyQT5 to plot the data.

The four violin strings are plotted:
- Blue: G
- Green: D
- Red: A
- Yellow: E

And the pitch is plotted and tracked in real time.

## Todo

Next goal is to integrate two modes of pitch correction:

### 1. Out of tune

Given a tuning standard (e.g., A=440Hz), provide users a notification within some threshold that they are out of tune, and by how much.

### 2. Wrong note

Given a YouTube link, convert to MIDI or MP3 and store as an array of pitches to compare user input to, and correct users if they play a wrong note - what they did play, and what they should have played.
