# violin-CV

**Current goal:** Real-time pitch correction

## PitchAnalyzer.py
(Can you guess the piece?)

<img width="300" alt="Screenshot 2024-06-05 at 11 05 32 AM" src="https://github.com/hyuncat/violin-CV/assets/114366569/3a1f7bfb-9e98-47b8-b573-83bfa1694c58">

Currently, the program uses the Essentia library to analyze pitch from audio input (returning Hz + condifence), and PyQT5 to plot the data.

The four violin strings are marked:
- Blue: G
- Green: D
- Red: A
- Yellow: E

And the recorded pitch is plotted and tracked in real time.

## Todo

Next goal is to integrate two modes of pitch correction:

### 1. Out of tune

Given a tuning standard (e.g., A=440Hz), provide users a notification within some threshold that they are out of tune, and by how much.

### 2. Wrong note

Given a YouTube link, convert to MIDI or MP3 and store as an array of pitches to compare user input to, and correct users if they play a wrong note - what they did play, and what they should have played.
