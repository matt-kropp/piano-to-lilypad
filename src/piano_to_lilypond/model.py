import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from .utils.midi_utils import build_vocab, token_to_id
from .config import ENC_FEAT_DIM, ENC_HIDDEN_DIM, NUM_ENCODER_LAYERS, \
    NUM_DECODER_LAYERS, NUM_HEADS, FFN_DIM, DROPOUT, VOCAB_SIZE, DEVICE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        # Create initial positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def _extend_pe(self, length):
        """Extend positional encoding to handle longer sequences."""
        if length <= self.pe.size(0):
            return
            
        # Create extended positional encoding
        pe = torch.zeros(length, self.d_model, device=self.pe.device, dtype=self.pe.dtype)
        position = torch.arange(0, length, dtype=torch.float, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=self.pe.device) * 
                           (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        
        # Update the buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [T, B, D]
        seq_len = x.size(0)
        
        # Extend positional encoding if needed
        if seq_len > self.pe.size(0):
            self._extend_pe(seq_len)
            
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class PianoTransformer(nn.Module):
    def __init__(self, vocab_size=None):
        super(PianoTransformer, self).__init__()
        # Build or load vocab
        build_vocab()
        self.vocab_size = vocab_size if vocab_size is not None else len(token_to_id)
        # Encoder: Conv1d stem + TransformerEncoder
        self.conv1 = nn.Conv1d(in_channels=229 * 5, out_channels=ENC_FEAT_DIM, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=ENC_FEAT_DIM, out_channels=ENC_FEAT_DIM, kernel_size=5, stride=2, padding=2)
        self.enc_input_proj = nn.Linear(ENC_FEAT_DIM, ENC_HIDDEN_DIM)
        self.pos_encoder = PositionalEncoding(ENC_HIDDEN_DIM, dropout=DROPOUT)
        encoder_layer = TransformerEncoderLayer(d_model=ENC_HIDDEN_DIM, nhead=NUM_HEADS, 
                                              dim_feedforward=FFN_DIM, dropout=DROPOUT, 
                                              batch_first=False, norm_first=True)  # norm_first for better training
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)

        # Decoder: Standard TransformerDecoder
        self.token_emb = nn.Embedding(self.vocab_size, ENC_HIDDEN_DIM)
        self.pos_decoder = PositionalEncoding(ENC_HIDDEN_DIM, dropout=DROPOUT)
        decoder_layer = TransformerDecoderLayer(d_model=ENC_HIDDEN_DIM, nhead=NUM_HEADS, 
                                              dim_feedforward=FFN_DIM, dropout=DROPOUT,
                                              batch_first=False, norm_first=True)  # norm_first for better training
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=NUM_DECODER_LAYERS)
        self.output_proj = nn.Linear(ENC_HIDDEN_DIM, self.vocab_size)

    def encode(self, src):
        # src: [B, 5, T, N_MELS] if stacking 5 frames; flatten to [B, (5*N_MELS), T]
        B, T, stack, M = src.shape  # reorder dims
        # combine stack*mel frequency
        x = src.view(B, T, stack * M).permute(0, 2, 1)  # [B, 5*229, T]
        x = self.conv1(x)  # [B, ENC_FEAT_DIM, T/2]
        x = F.gelu(x)
        x = self.conv2(x)  # [B, ENC_FEAT_DIM, T/4]
        x = F.gelu(x)
        x = x.permute(2, 0, 1)  # [T/4, B, ENC_FEAT_DIM]
        x = self.enc_input_proj(x)  # [T/4, B, ENC_HIDDEN_DIM]
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)  # [T/4, B, ENC_HIDDEN_DIM]
        return memory

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: [L, B] token indices
        embedded = self.token_emb(tgt) * (ENC_HIDDEN_DIM ** 0.5)  # [L, B, D]
        embedded = self.pos_decoder(embedded)
        output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.output_proj(output)  # [L, B, vocab_size]
        return logits

    def forward(self, src, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        memory = self.encode(src)
        logits = self.decode(tgt, memory, tgt_mask, tgt_key_padding_mask)
        return logits