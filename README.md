# Piano to LilyPond

A deep learning system that converts piano audio recordings to LilyPond sheet music notation using a Transformer-based architecture.

## Overview

This project implements an end-to-end system that:
1. Takes piano audio (WAV files) as input
2. Converts audio to mel-spectrograms with temporal context
3. Uses a Transformer encoder-decoder to generate musical tokens
4. Converts tokens to LilyPond notation for sheet music rendering

## Architecture

- **Encoder**: Convolutional layers + Transformer encoder for audio feature extraction
- **Decoder**: Transformer decoder for sequential token generation
- **Vocabulary**: 818 tokens including notes, velocities, timing, pedal, and voice assignments
- **Context**: 5-frame temporal stacking for improved audio representation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd piano-to-lilypad
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training

Train the model on MAESTRO dataset:
```bash
p2l_train
```

The training script expects audio-MIDI pairs in the `./data/maestro/` directory.

### Inference

Convert piano audio to LilyPond notation:
```bash
p2l_infer --audio input.wav --out output.ly --checkpoint model.ckpt --tempo 120.0
```

Parameters:
- `--audio`: Path to input WAV file
- `--out`: Path to output LilyPond (.ly) file
- `--checkpoint`: Path to trained model checkpoint
- `--tempo`: Estimated tempo for conversion (default: 120.0 BPM)

## Configuration

Key configuration parameters in `src/piano_to_lilypond/config.py`:

- **Audio**: 44.1kHz sample rate, 229 mel bands
- **Model**: 12 encoder/decoder layers, 8 attention heads
- **Training**: Batch size 16, learning rate 1e-4
- **Vocabulary**: 818 tokens (notes, timing, dynamics, voices)

## Data Format

The system expects:
- **Audio**: WAV files at 44.1kHz sample rate
- **MIDI**: Corresponding MIDI files for training
- **Output**: LilyPond (.ly) files for sheet music rendering

## Token Vocabulary

The system uses a comprehensive token vocabulary:
- **Note events**: `NOTE_ON_{pitch}_{velocity}`, `NOTE_OFF_{pitch}`
- **Timing**: `TS_{1-100}` (10ms time shifts)
- **Voices**: `VOICE_{0-3}` (4-voice polyphony)
- **Pedal**: `PEDAL_ON`, `PEDAL_OFF`
- **Dynamics**: `DYN_{pp,p,mp,mf,f,ff}`
- **Special**: `EOS` (end of sequence), `PAD` (padding)

## Fixes Applied

The following critical issues were identified and fixed:

1. **Import Order Error** (`dataset.py`): Fixed `build_vocab()` being called before import
2. **Syntax Error** (`infer.py`): Removed invalid triple backticks
3. **Vocabulary Access** (`train.py`): Fixed incorrect vocabulary dictionary access
4. **Entry Points** (`setup.py`): Corrected module paths for console scripts
5. **String Formatting** (`lilypond_utils.py`): Fixed multiline string syntax error
6. **Vocabulary Initialization**: Added automatic vocabulary building on module import
7. **Sequence Length Error** (`model.py`): Fixed positional encoding to handle sequences longer than 10,000 frames
8. **Memory Issues** (`dataset.py`): Added audio sequence length limiting to prevent out-of-memory errors

## Dependencies

- PyTorch >= 1.12.0
- librosa >= 0.9.2
- pretty_midi >= 0.2.10
- soundfile >= 0.12.1
- numpy >= 1.21.0
- scipy >= 1.7.0

## Project Structure

```
src/piano_to_lilypond/
├── __init__.py
├── config.py          # Configuration parameters
├── model.py           # Transformer model architecture
├── dataset.py         # Data loading and preprocessing
├── train.py           # Training script
├── infer.py           # Inference script
└── utils/
    ├── audio_utils.py     # Audio processing utilities
    ├── midi_utils.py      # MIDI tokenization utilities
    └── lilypond_utils.py  # LilyPond conversion utilities
```

## Notes

- The model uses EOS token as both start-of-sequence and end-of-sequence marker
- Audio is processed with 5-frame temporal context for better feature representation
- The system supports 4-voice polyphonic transcription
- LilyPond output includes voice assignments, pedal markings, and basic dynamics

## License

[Add your license information here] 