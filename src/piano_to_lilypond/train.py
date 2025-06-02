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

def check_memory():
    """Check system and GPU memory usage."""
    # System memory
    memory = psutil.virtual_memory()
    print(f"System Memory: {memory.percent:.1f}% used ({memory.available/1024**3:.1f}GB available)")
    
    # GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_memory_max:.2f}GB max, {gpu_memory_reserved:.2f}GB reserved")
        
        # Warning if memory usage is high
        if memory.percent > 85:
            print("⚠️  WARNING: System memory usage is high!")
        if gpu_memory_reserved > 0.8 * torch.cuda.get_device_properties(0).total_memory / 1024**3:
            print("⚠️  WARNING: GPU memory usage is high!")

# Simple function to gather (audio, midi) file pairs from MAESTRO directory

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


def create_optimizer_and_scheduler(model):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        return max(
            0.0,
            float(TOTAL_STEPS - step) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
        )
    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, scaler):
    model.train()
    total_loss = 0.0
    accumulation_loss = 0.0
    successful_steps = 0
    
    # Clear GPU cache at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    for step, (src, tgt) in enumerate(dataloader):
        try:
            # Emergency memory check - if we're using too much, clear cache
            if torch.cuda.is_available() and step % 10 == 0:
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                if memory_used > 30:  # Increased from 20GB - A100 has 40GB
                    print(f"High memory usage detected: {memory_used:.1f}GB, clearing cache...")
                    torch.cuda.empty_cache()
            
            src = src.to(DEVICE, non_blocking=True)  # [B, T, 5, N_MELS] - non_blocking for GPU
            tgt = tgt.to(DEVICE, non_blocking=True)  # [B, L]
            
            # Dataset already handles length limiting, no need for double-check here
            
            # Prepare input to decoder: shift right, feed <EOS> at start
            tgt_input = torch.cat([torch.full((tgt.size(0), 1), token_to_id['EOS'], dtype=torch.long, device=DEVICE), tgt[:, :-1]], dim=1)
            # Create tgt mask
            tgt_len = tgt_input.size(1)
            tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=DEVICE) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

            # Use autocast for mixed precision training on GPU
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    logits = model(src, tgt_input.transpose(0,1), tgt_mask=tgt_mask)
                    # logits: [L, B, Vocab]
                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=token_to_id['PAD'])
                    loss = loss_fn(logits.view(-1, model.vocab_size), tgt.transpose(0,1).reshape(-1))
            else:
                logits = model(src, tgt_input.transpose(0,1), tgt_mask=tgt_mask)
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=token_to_id['PAD'])
                loss = loss_fn(logits.view(-1, model.vocab_size), tgt.transpose(0,1).reshape(-1))
            
            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            accumulation_loss += loss.item()
            
            # Use scaler for mixed precision if available
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            successful_steps += 1
            
            # Only update weights every GRADIENT_ACCUMULATION_STEPS
            if successful_steps % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += accumulation_loss
                if (successful_steps // GRADIENT_ACCUMULATION_STEPS) % 25 == 0:  # More frequent logging for A100
                    print(f"Epoch {epoch}, Step {successful_steps//GRADIENT_ACCUMULATION_STEPS}, Loss: {accumulation_loss:.4f}")
                accumulation_loss = 0.0
                
                # Less frequent cache clearing for GPU
                if torch.cuda.is_available() and successful_steps % (GRADIENT_ACCUMULATION_STEPS * 50) == 0:
                    torch.cuda.empty_cache()
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: Out of memory at step {step}, clearing cache and skipping batch")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                # Reduce batch size dynamically if we keep hitting OOM
                if hasattr(train_one_epoch, 'oom_count'):
                    train_one_epoch.oom_count += 1
                else:
                    train_one_epoch.oom_count = 1
                if train_one_epoch.oom_count > 5:
                    print("Too many OOM errors, consider reducing batch size or model size further")
                continue
            else:
                raise e
        except Exception as e:
            print(f"WARNING: Error processing batch at step {step}: {e}")
            continue
    
    avg_loss = total_loss / max(1, successful_steps // GRADIENT_ACCUMULATION_STEPS)
    print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}, Successful steps: {successful_steps}")
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


def main():
    print("🚀 Starting Piano to LilyPond Training")
    print("=" * 50)
    
    # Check system resources
    check_memory()
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print("=" * 50)
    
    # Build dataset
    maestro_pairs = get_file_list(MAESTRO_DIR)
    print(f"Found {len(maestro_pairs)} audio-MIDI pairs")
    
    if len(maestro_pairs) == 0:
        print("❌ No training data found! Please check MAESTRO_DIR path.")
        return
    
    # Set fixed random seed for consistent train/val split (important for caching)
    random.seed(42)
    random.shuffle(maestro_pairs)
    split = int(0.95 * len(maestro_pairs))
    train_pairs = maestro_pairs[:split]
    val_pairs = maestro_pairs[split:]
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Cache directory: {os.path.join(os.getcwd(), '.dataset_cache')}")
    
    print("🗂️ Creating training dataset...")
    train_dataset = PianoDataset(train_pairs, max_seq_len=MAX_MIDI_LENGTH)
    print("🗂️ Creating validation dataset...")
    val_dataset = PianoDataset(val_pairs, max_seq_len=MAX_MIDI_LENGTH)
    
    # Use multiple workers for faster data loading on GPU (reduced to prevent memory issues)
    num_workers = 2 if torch.cuda.is_available() else 0  # Reduced from 4 to 2
    pin_memory = torch.cuda.is_available()
    
    try:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                collate_fn=PianoDataset.collate_fn, num_workers=num_workers, 
                                pin_memory=pin_memory, persistent_workers=num_workers > 0,
                                prefetch_factor=2 if num_workers > 0 else None)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              collate_fn=PianoDataset.collate_fn, num_workers=num_workers,
                              pin_memory=pin_memory, persistent_workers=num_workers > 0,
                              prefetch_factor=2 if num_workers > 0 else None)
    except Exception as e:
        print(f"Warning: Failed to create DataLoader with {num_workers} workers: {e}")
        print("Falling back to single-threaded data loading...")
        num_workers = 0
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                collate_fn=PianoDataset.collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              collate_fn=PianoDataset.collate_fn, num_workers=0)

    print("✅ Datasets created successfully")
    check_memory()

    # Initialize model
    print("Creating model...")
    model = PianoTransformer().to(DEVICE)
    print(f"Model vocabulary size: {model.vocab_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer, scheduler = create_optimizer_and_scheduler(model)
    start_step = 0
    
    print("✅ Model created successfully")
    check_memory()
    
    # Create mixed precision scaler for GPU training
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Optionally, load from checkpoint
    # if os.path.exists(os.path.join(CHECKPOINT_DIR, 'latest.ckpt')):
    #     ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'latest.ckpt'))
    #     model.load_state_dict(ckpt['model_state_dict'])
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #     scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    #     start_step = ckpt['step']

    print("🎵 Starting training...")
    for epoch in range(1, 11):
        print(f"\n📖 Epoch {epoch}/10")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, scaler)
        # TODO: validate on val_loader, compute metrics
        # Save checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch}.ckpt")
        save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")
        check_memory()

if __name__ == '__main__':
    main()