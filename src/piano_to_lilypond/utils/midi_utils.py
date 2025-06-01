import pretty_midi
import numpy as np

# Define token mapping global variables
PITCHES = list(range(21, 109))  # A0 to C8 inclusive
VELOCITY_BINS = [0, 20, 40, 60, 80, 100, 120, 128]
TIME_SHIFT_BINS = list(range(1, 101))  # Each = 10ms up to 1s
VOICES = [0, 1, 2, 3]  # Four voices
DYNAMICS = ["pp", "p", "mp", "mf", "f", "ff"]

# Build vocab dictionaries (to be populated once)
token_to_id = {}
id_to_token = {}


def build_vocab():
    global token_to_id, id_to_token
    if token_to_id:  # Already built
        return token_to_id, id_to_token
        
    token_to_id = {}
    id_to_token = {}
    idx = 0
    # Note-On tokens: format "NOTE_ON_{pitch}_{vbin}"
    for p in PITCHES:
        for i, v in enumerate(VELOCITY_BINS[:-1]):
            token = f"NOTE_ON_{p}_{i}"
            token_to_id[token] = idx; id_to_token[idx] = token; idx += 1
    # Note-Off tokens: "NOTE_OFF_{pitch}"
    for p in PITCHES:
        token = f"NOTE_OFF_{p}"
        token_to_id[token] = idx; id_to_token[idx] = token; idx += 1
    # Pedal tokens
    for name in ["PEDAL_ON", "PEDAL_OFF"]:
        token_to_id[name] = idx; id_to_token[idx] = name; idx += 1
    # Dynamic tokens
    for dyn in DYNAMICS:
        token = f"DYN_{dyn}"
        token_to_id[token] = idx; id_to_token[idx] = token; idx += 1
    # Voice assign tokens
    for v in VOICES:
        token = f"VOICE_{v}"
        token_to_id[token] = idx; id_to_token[idx] = token; idx += 1
    # Time-shift tokens: "TS_{k}" (k from 1 to 100)
    for k in TIME_SHIFT_BINS:
        token = f"TS_{k}"
        token_to_id[token] = idx; id_to_token[idx] = token; idx += 1
    # Special tokens
    token_to_id["EOS"] = idx; id_to_token[idx] = "EOS"; idx += 1
    token_to_id["PAD"] = idx; id_to_token[idx] = "PAD"; idx += 1
    return token_to_id, id_to_token


# Build vocabulary when module is imported
build_vocab()


def midi_to_token_sequence(midi_path, tempo=None):
    """
    Convert a MIDI file to a list of tokens.
    Quantize time to 10ms bins and assign voices via simple heuristic.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    if tempo is None:
        # Estimate tempo from MIDI or default to 120 BPM
        try:
            tempo_changes, tempi = pm.get_tempo_changes()
            tempo = tempi[0]
        except:
            tempo = 120.0
    events = []  # list of (time_in_sec, event_type, params)

    # Pedal events (controller 64 = sustain pedal)
    for control in pm.instruments[0].control_changes:
        if control.number == 64:
            if control.value >= 64:
                events.append((control.time, 'PEDAL_ON', {}))
            else:
                events.append((control.time, 'PEDAL_OFF', {}))

    # Note events
    for inst in pm.instruments:
        for note in inst.notes:
            if note.pitch in PITCHES:
                events.append((note.start, 'NOTE_ON', {'pitch': note.pitch, 'velocity': note.velocity}))
                events.append((note.end, 'NOTE_OFF', {'pitch': note.pitch}))

    # Sort events by time
    events.sort(key=lambda x: x[0])

    # Voice assignment: simple split by pitch
    def assign_voice(pitch):
        if pitch >= 60:
            return 0 if pitch >= 72 else 1
        else:
            return 2 if pitch < 48 else 3

    # Build token sequence with TimeShift
    token_seq = []
    prev_time = 0.0
    for (t, etype, params) in events:
        delta = t - prev_time
        n_bins = int(round(delta * 100))  # each bin = 0.01 s
        while n_bins > 0:
            step = min(n_bins, 100)
            token_seq.append(f"TS_{step}")
            n_bins -= step
        if etype == 'NOTE_ON':
            pitch = params['pitch']
            vel = params['velocity']
            # assign velocity bin
            v_idx = np.digitize(vel, VELOCITY_BINS) - 1
            token_seq.append(f"VOICE_{assign_voice(pitch)}")
            token_seq.append(f"NOTE_ON_{pitch}_{v_idx}")
        elif etype == 'NOTE_OFF':
            pitch = params['pitch']
            token_seq.append(f"VOICE_{assign_voice(pitch)}")
            token_seq.append(f"NOTE_OFF_{pitch}")
        elif etype == 'PEDAL_ON':
            token_seq.append("PEDAL_ON")
        elif etype == 'PEDAL_OFF':
            token_seq.append("PEDAL_OFF")
        prev_time = t
    token_seq.append("EOS")
    return token_seq
