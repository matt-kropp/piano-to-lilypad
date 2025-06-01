import os
import torch
import argparse
from .model import PianoTransformer
from .utils.audio_utils import load_audio, compute_mel_spectrogram
from .utils.midi_utils import build_vocab, token_to_id, id_to_token
from .utils.lilypond_utils import tokens_to_lilypond
from .config import SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH, DEVICE, BEAM_SIZE, MAX_DECODING_STEPS


def greedy_decode(model, src, sos_id, eos_id, max_len=10000):
    """
    Greedy decoding (no beam search): returns a list of token IDs.
    """
    model.eval()
    memory = model.encode(src.unsqueeze(0).to(DEVICE))  # [T, 1, D]
    ys = torch.tensor([[sos_id]], dtype=torch.long, device=DEVICE)
    for i in range(max_len):
        tgt_mask = (torch.triu(torch.ones((ys.size(0), ys.size(0)), device=DEVICE)) == 1).transpose(0,1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        out = model.decode(ys, memory, tgt_mask=tgt_mask)
        prob = out[-1].softmax(dim=-1)
        next_id = torch.argmax(prob, dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_id]], device=DEVICE)], dim=0)
        if next_id == eos_id:
            break
    return ys.squeeze(1).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True, help='Path to input WAV file')
    parser.add_argument('--out', type=str, required=True, help='Path to output .ly file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tempo', type=float, default=120.0, help='Estimated tempo for LilyPond conversion')
    args = parser.parse_args()

    build_vocab()
    sos_id = token_to_id['EOS']
    eos_id = token_to_id['EOS']

    # Load model
    model = PianoTransformer().to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    # Load and preprocess audio
    wav = load_audio(args.audio, SAMPLE_RATE)
    mel = compute_mel_spectrogram(wav, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH)
    T = mel.shape[1]
    # Stack ±2 frames for context
    stacked = []
    stack_size = 5
    for t in range(T):
        frames = []
        for offset in range(-(stack_size//2), (stack_size//2)+1):
            tt = min(max(t + offset, 0), T - 1)
            frames.append(mel[:, tt])
        stacked.append(torch.stack([torch.from_numpy(f) for f in frames], dim=0))
    src = torch.stack(stacked, dim=0).float()  # [T, 5, N_MELS]

    # Decode tokens
    token_ids = greedy_decode(model, src, sos_id, eos_id)
    token_seq = [id_to_token[i] for i in token_ids]

    # Convert tokens to LilyPond
    tokens_to_lilypond(token_seq, args.out, tempo=args.tempo)
    print(f"LilyPond file written to {args.out}")

if __name__ == '__main__':
    main()