import os
import math
import torch
import random 
import psutil
import gc
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from .dataset import PianoDataset
from .model import PianoTransformer
from .config import (
    MAESTRO_DIR, SYNTHTRAIN_DIR, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, WARMUP_STEPS, TOTAL_STEPS, AUX_CTC_WEIGHT, CHECKPOINT_DIR, DEVICE,
    GRADIENT_ACCUMULATION_STEPS, MAX_MIDI_LENGTH, MAX_AUDIO_LENGTH
)
from .utils.midi_utils import build_vocab, id_to_token, token_to_id

def get_file_list(maestro_dir):
    pairs = []
    for root, _, files in os.walk(maestro_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                base = f[:-4]
                wav_path = os.path.join(root, f)
                mid_path = os.path.join(root, base + '.midi')
                if os.path.exists(mid_path):
                    pairs.append((wav_path, mid_path))
    return pairs

def main():
    print("🚀 Starting Piano to LilyPond Training - SUBSET MODE")
    print("=" * 60)
    print("This script trains on 10% of data for quick experimentation")
    print("=" * 60)
    
    # Get all pairs
    maestro_pairs = get_file_list(MAESTRO_DIR)
    print(f"Found {len(maestro_pairs)} total audio-MIDI pairs")
    
    # Use only 10% for quick experimentation
    random.seed(42)
    random.shuffle(maestro_pairs)
    subset_size = max(20, len(maestro_pairs) // 10)  # At least 20 files
    maestro_pairs = maestro_pairs[:subset_size]
    
    print(f"🎯 Using SUBSET of {len(maestro_pairs)} pairs for experimentation")
    print(f"📊 Batch size: {BATCH_SIZE} (was 12)")
    print(f"📊 Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} (was 4)")
    print(f"📊 Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    
    if len(maestro_pairs) == 0:
        print("❌ No training data found! Please check MAESTRO_DIR path.")
        return
    
    # Split subset
    split = int(0.8 * len(maestro_pairs))  # 80/20 split for subset
    train_pairs = maestro_pairs[:split]
    val_pairs = maestro_pairs[split:]
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    
    # Create datasets
    print("🗂️ Creating training dataset...")
    train_dataset = PianoDataset(train_pairs, max_seq_len=MAX_MIDI_LENGTH)
    print("🗂️ Creating validation dataset...")
    val_dataset = PianoDataset(val_pairs, max_seq_len=MAX_MIDI_LENGTH)
    
    # Data loaders
    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=PianoDataset.collate_fn, num_workers=num_workers, 
                            pin_memory=pin_memory, persistent_workers=num_workers > 0,
                            prefetch_factor=2 if num_workers > 0 else None)
    
    print(f"✅ Subset datasets created: {len(train_dataset)} train windows, {len(val_dataset)} val windows")
    
    # Check memory before model creation
    if torch.cuda.is_available():
        print(f"🧠 GPU Memory before model: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    # Initialize model
    print("🏗️ Creating larger model...")
    model = PianoTransformer().to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    if torch.cuda.is_available():
        print(f"🧠 GPU Memory after model: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        return max(0.0, float(TOTAL_STEPS - step) / float(max(1, TOTAL_STEPS - WARMUP_STEPS)))
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    print(f"🎵 Starting subset training (3 epochs for quick validation)...")
    
    # Quick 3-epoch training
    for epoch in range(1, 4):
        print(f"\n📖 Epoch {epoch}/3")
        model.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        
        print(f"📊 Processing {total_batches} batches...")
        
        for step, (src, tgt) in enumerate(train_loader):
            try:
                if step % 10 == 0:
                    progress = step / total_batches * 100
                    print(f"  📈 Batch {step+1}/{total_batches} ({progress:.1f}%)")
                    if torch.cuda.is_available():
                        gpu_mem = torch.cuda.memory_allocated() / 1024**3
                        print(f"  🧠 GPU Memory: {gpu_mem:.1f}GB")
                
                src = src.to(DEVICE, non_blocking=True)
                tgt = tgt.to(DEVICE, non_blocking=True)
                
                # Prepare decoder input
                tgt_input = torch.cat([torch.full((tgt.size(0), 1), token_to_id['EOS'], 
                                                dtype=torch.long, device=DEVICE), tgt[:, :-1]], dim=1)
                tgt_len = tgt_input.size(1)
                tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=DEVICE) == 1).transpose(0, 1)
                tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
                
                # Forward pass with mixed precision
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        logits = model(src, tgt_input.transpose(0,1), tgt_mask=tgt_mask)
                        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=token_to_id['PAD'])
                        loss = loss_fn(logits.view(-1, model.vocab_size), tgt.transpose(0,1).reshape(-1))
                else:
                    logits = model(src, tgt_input.transpose(0,1), tgt_mask=tgt_mask)
                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=token_to_id['PAD'])
                    loss = loss_fn(logits.view(-1, model.vocab_size), tgt.transpose(0,1).reshape(-1))
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                
                optimizer.zero_grad()
                total_loss += loss.item()
                
                if step % 10 == 0:
                    print(f"    💰 Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ OOM at batch {step+1} - try reducing batch size further")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch} complete | Average Loss: {avg_loss:.4f}")
    
    print(f"\n🎉 Subset training completed successfully!")
    print(f"📊 Final GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.1f}GB / 40GB")
    print(f"💡 If this works well, scale up to full dataset with train.py")

if __name__ == '__main__':
    main() 