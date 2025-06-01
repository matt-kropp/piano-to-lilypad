import os
from .midi_utils import id_to_token

# Basic pitch name mapping: MIDI number -> LilyPond pitch (in C major reference)
NOTE_NAMES = ['c', 'cis', 'd', 'dis', 'e', 'f', 'fis', 'g', 'gis', 'a', 'ais', 'b']

def midi_to_lily_pitch(pitch):
    octave = (pitch // 12) - 1  # MIDI octave: C4=60 => octave=4
    name = NOTE_NAMES[pitch % 12]
    # LilyPond uses c' = middle C (C4). So octave offsets:
    lily = name
    if octave > 3:
        lily += "'" * (octave - 3)
    elif octave < 3:
        lily += "," * (3 - octave)
    return lily


def duration_to_lily_dur(frac):
    # frac = fraction of whole note (e.g. quarter=1/4 => 0.25)
    # Convert to nearest standard note or dotted note: 1, 2, 4, 8, 16, 32, 64
    # Example: 0.25 -> 4; 0.125 -> 8; 0.375 -> 8.
    denom = int(round(1 / frac))
    if denom in [1, 2, 4, 8, 16, 32, 64]:
        return str(denom)
    # Check dotted (e.g. 6 -> dotted quarter)
    for base in [1, 2, 4, 8, 16, 32]:
        if abs(frac - (1 / base) * 1.5) < 1e-3:
            return f"{base}."
    # Fallback to nearest power-of-two
    nearest = min([1, 2, 4, 8, 16, 32, 64], key=lambda x: abs(frac - 1 / x))
    return str(nearest)


def tokens_to_lilypond(token_seq, output_path, tempo=120):
    """
    Convert a token sequence (from Stage I) back into a .ly file.
    This is a simplified deterministic converter.
    """
    # Group tokens into events with approximate measure placement
    # First, parse the token sequence back to time-annotated events
    events = []  # list of (time_bin_idx, token)
    time_acc = 0
    for tok in token_seq:
        if tok.startswith("TS_"):
            step = int(tok.split("_")[1])
            time_acc += step  # each step = 10ms
        else:
            events.append((time_acc, tok))
    # Convert time_acc (in 10ms bins) to beats: 10ms = 0.01s. At 120 BPM, 1 beat = 0.5s = 50 bins
    bin2beat = 0.01 / (60.0 / tempo)

    # Build measures: assume 4/4 time
    bins_per_beat = 60.0 / tempo / 0.01  # ~50 for 120 BPM
    bins_per_measure = bins_per_beat * 4

    measures_RH = []  # list of list of note events for RH voices
    measures_LH = []
    current_measure_RH = []
    current_measure_LH = []

    # Keep track of onsets for assembling durations
    active_notes = {}  # (voice, pitch) -> (start_bin)

    for (bin_idx, tok) in events:
        if tok == "EOS":
            break
        if tok.startswith("VOICE_"):
            voice = int(tok.split("_")[1])
            continue
        if tok.startswith("NOTE_ON_"):
            _, p, vbin = tok.split("_")
            pitch = int(p)
            vel_idx = int(vbin)
            active_notes[(voice, pitch)] = bin_idx
        elif tok.startswith("NOTE_OFF_"):
            _, p = tok.split("_")
            pitch = int(p)
            start_bin = active_notes.pop((voice, pitch), None)
            if start_bin is None:
                continue
            duration_bins = bin_idx - start_bin
            # Determine duration fraction in quarters: dur_beats = duration_bins / bins_per_beat
            dur_beats = duration_bins / bins_per_beat
            frac = dur_beats / 4.0  # fraction of whole note
            dur_token = duration_to_lily_dur(frac)
            lily_pitch = midi_to_lily_pitch(pitch)
            lily_note = f"{lily_pitch}{dur_token}"
            if voice in [0, 1]:
                current_measure_RH.append((voice, lily_note))
            else:
                current_measure_LH.append((voice, lily_note))
        elif tok == "PEDAL_ON":
            # Represent as \sustainOn
            if voice in [0, 1]:
                current_measure_RH.append((voice, "\\sustainOn"))
            else:
                current_measure_LH.append((voice, "\\sustainOn"))
        elif tok == "PEDAL_OFF":
            if voice in [0, 1]:
                current_measure_RH.append((voice, "\\sustainOff"))
            else:
                current_measure_LH.append((voice, "\\sustainOff"))
        # Check if bin_idx crosses measure boundary
        if bin_idx >= bins_per_measure * (len(measures_RH) + 1):
            measures_RH.append(current_measure_RH)
            measures_LH.append(current_measure_LH)
            current_measure_RH = []
            current_measure_LH = []
    # Append final partial measure
    if current_measure_RH or current_measure_LH:
        measures_RH.append(current_measure_RH)
        measures_LH.append(current_measure_LH)

    # Generate LilyPond content
    lines = []
    lines.append("\\version \"2.24.1\"")
    lines.append("\\header { title = \"Transcription\" }")
    lines.append("\\layout { indent = 0\\mm }")
    lines.append("\\score { <<")
    lines.append("  \\new PianoStaff <<")
    # Right hand staff
    lines.append("    \\new Staff = \"right\" { \\\\clef treble \\\\time 4/4 ")
    for measure in measures_RH:
        if measure:
            # Insert voice markers
            for (v, note) in measure:
                if v == 0:
                    lines.append("      \\voiceOne " + note)
                elif v == 1:
                    lines.append("      \\voiceTwo " + note)
            lines.append("      | ")
        else:
            lines.append("      | ")
    lines.append("    }  ")
    # Left hand staff
    lines.append("    \\new Staff = \"left\" { \\\\clef bass \\\\time 4/4 ")
    for measure in measures_LH:
        if measure:
            for (v, note) in measure:
                if v == 2:
                    lines.append("      \\voiceOne " + note)
                elif v == 3:
                    lines.append("      \\voiceTwo " + note)
            lines.append("      | ")
        else:
            lines.append("      | ")
    lines.append("    }  ")
    lines.append("  >>")
    lines.append(">>")
    lines.append("  \\midi {}")
    lines.append("}")

    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))