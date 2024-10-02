import numpy as np

def midi_to_freq(midi_num: float, tuning: float=440.0) -> float:
    """
    Convert a MIDI note number to frequency.
    """
    return tuning * (2 ** ((midi_num-69) / 12))

def freq_to_midi(freq: float, tuning: float=440.0) -> float:
    """
    Convert a frequency to a MIDI note number.
    """
    return 69 + 12 * np.log2(freq / tuning)

def freq_to_pitchbin(freq: float, bins_per_semitone: int=10, tuning: float=440.0, fmin=196, fmax=5000) -> int:
    """
    Convert a frequency to a pitch bin.
    """
    midi_num = freq_to_midi(freq, tuning)
    pitch_bin = int(np.round(midi_num * bins_per_semitone))
    # Make sure its within range of freq 196 - 5000
    min_midi = freq_to_midi(fmin, tuning)
    max_midi = freq_to_midi(fmax, tuning)
    return int(np.clip(pitch_bin, min_midi * bins_per_semitone, max_midi * bins_per_semitone))