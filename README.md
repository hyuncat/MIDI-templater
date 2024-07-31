# MIDI-templater

Load a MIDI file, an audio file, and see/playback the differences!

<img width="600" alt="app_preview" src="https://github.com/user-attachments/assets/e13cbf59-b5df-488e-8877-294deea74fa7">

### About the graph
- Treble clef staff lines are in black - the demo MIDI/audio files are for violin :-)
- MIDI notes are in black
- The audio file is parsed to detect all pitches, plotted in viridis color corresponding to the confidence of the pitch estimate.
- Note segmentation is estimated using a rolling median and is displayed in purple underneath each new detected note.

### Currently working on...
Improving accuracy of onset-matching in dynamic time warping by retrieving the different-enough note onsets as 'annotations'.
