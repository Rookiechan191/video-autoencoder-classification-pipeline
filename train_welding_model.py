import os
import gc
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models.video as video_models
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available, using CPU fallback.")


# ===========================================
# STEP 1: CONFIGURATION
# ===========================================
# !! Update these paths for your local machine !!
DATA_ROOT = r"C:\Users\Harshith\Downloads\intel_robotic_welding_dataset\raid\intel_robotic_welding_dataset"
MANIFEST_PATH = os.path.join(DATA_ROOT, "manifest.csv")
CHECKPOINT_DIR = os.path.join(DATA_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\nData root set to: {DATA_ROOT}")
print(f"Manifest path: {MANIFEST_PATH}")
print(f"Checkpoints will be saved in: {CHECKPOINT_DIR}")

# ===========================================
# STEP 2: DATASET CLASS (Improved with logging)
# ===========================================
class WeldVideoManifestDataset(Dataset):
    def __init__(self, manifest_df, data_root='', window_size=64, stride=None):
        self.video_paths = []
        # MODIFIED: Added check for manifest_df
        if manifest_df is None or manifest_df.empty:
            print("Warning: Manifest DataFrame is empty.")
            return

        for _, row in manifest_df.iterrows():
            subdir_path = os.path.join(data_root, row['SUBDIRS'])
            subdir_name = os.path.basename(row['SUBDIRS'])
            video_file = os.path.join(subdir_path, f"{subdir_name}.avi")
            if os.path.exists(video_file):
                self.video_paths.append(video_file)
            else:
                print(f" Missing video file: {video_file}")
                
        self.window_size = window_size
        self.stride = stride if stride else window_size
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor()
        ])
        
        # MODIFIED: Added more robust logging from Code 1
        if len(self.video_paths) == 0:
            print("Warning: No videos found for this dataset chunk.")
        else:
            print(f" Dataset initialized with {len(self.video_paths)} videos.")

    def __len__(self):
        return len(self.video_paths)

    def get_video_clips(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(frame_rgb))
            except cv2.error as e:
                print(f"Warning: Skipping corrupt frame in {video_path}. Error: {e}")
                continue
        cap.release()
        
        # This check from Code 2 is already good
        if len(frames) < self.window_size:
            return [] 
            
        clips = []
        for i in range(0, len(frames) - self.window_size + 1, self.stride):
            clip = torch.stack(frames[i:i+self.window_size], dim=1)
            clips.append(clip)
        return clips

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        clips = self.get_video_clips(video_path)
        return clips, video_path

# ===========================================
# STEP 3: MODEL DEFINITIONS
# ===========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Pretrained feature extractor
try:
    feature_net = video_models.r3d_18(weights="KINETICS400_V1").to(device)
except Exception:
    feature_net = video_models.r3d_18(pretrained=True).to(device)
feature_net.eval()
print("Feature extractor (r3d_18) loaded.")

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=400, bottleneck_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

autoencoder = AutoEncoder().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=5e-4)
criterion_recon = nn.SmoothL1Loss()
latent_reg_weight = 1e-4
print("Autoencoder model created.")
print("Optimizer (Adam) and Loss (SmoothL1 + L2 Latent) defined.")


# ===========================================
# STEP 4: FEATURE EXTRACTION FUNCTION
# ===========================================
@torch.no_grad()
def extract_features_batch(video_clips, batch_size=8):
    features_list = []
    for i in range(0, len(video_clips), batch_size):
        batch_clips = torch.stack(video_clips[i:i+batch_size], dim=0).to(device)
        batch_features = feature_net(batch_clips)
        features_list.append(batch_features.cpu())
        del batch_clips
        torch.cuda.empty_cache()
    if not features_list:
        return torch.empty(0, 400)
    return torch.cat(features_list, dim=0)

# ===========================================
# STEP 5: PIPELINE SETUP & CHECKPOINT LOADING
# ===========================================

# --- Hyperparameters ---
NUM_EPOCHS = 10
VIDEO_CHUNK_SIZE = 1
FEATURE_BATCH_SIZE = 8
AE_BATCH_SIZE = 64
DATASET_STRIDE = 128
DATASET_WINDOW = 16

# --- Checkpointing & Early Stopping (from Code 1) ---
LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
patience = 20
min_delta = 3e-4
best_loss = float('inf')
early_stop_counter = 0
start_epoch = 0

# --- Checkpoint Loading Logic (MODIFIED to be more robust than Code 1) ---
if os.path.exists(LATEST_CKPT_PATH):
    print(f"Loading latest checkpoint from {LATEST_CKPT_PATH}")
    try:
        checkpoint = torch.load(LATEST_CKPT_PATH, map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('loss', float('inf')) # Get last loss, or default
        print(f"Resuming training from Epoch {start_epoch + 1}...")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        start_epoch = 0
        best_loss = float('inf')
elif os.path.exists(BEST_MODEL_PATH):
    # Fallback if only the best model exists
    print(f"Found best model at {BEST_MODEL_PATH}, but no full checkpoint.")
    print("Loading model weights only and starting from epoch 0.")
    try:
        autoencoder.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading best model: {e}. Starting from scratch.")
    start_epoch = 0
    best_loss = float('inf') # We don't know the best loss, so we must reset it
else:
    print("No checkpoint found. Starting from scratch.")
    start_epoch = 0
    best_loss = float('inf')


# --- Load and Chunk Manifest ---
print("\nLoading and chunking manifest...")
try:
    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df['SPLIT'] == 'TRAIN'].reset_index(drop=True)
    if len(train_df) == 0:
        raise ValueError("No TRAIN split found in manifest.csv")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Manifest file not found at {MANIFEST_PATH}")
    sys.exit(1) # Exit the script
except Exception as e:
    print(f"Manifest loading failed: {e}")
    sys.exit(1) # Exit the script

num_chunks = max(1, len(train_df) // VIDEO_CHUNK_SIZE)
chunks = np.array_split(train_df, num_chunks)
print(f"Loaded {len(train_df)} TRAIN videos. Split into {len(chunks)} chunks of ~{VIDEO_CHUNK_SIZE} videos each.\n")


# ===========================================
# STEP 6: TRAINING LOOP (Upgraded)
# ===========================================
for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\n========== Epoch {epoch+1}/{NUM_EPOCHS} ==========")
    autoencoder.train()
    epoch_total_loss = 0.0
    epoch_total_samples = 0

    for ci, cdf in enumerate(chunks):
        print(f"\n Processing chunk {ci+1}/{len(chunks)} ({len(cdf)} videos)")
        dataset = WeldVideoManifestDataset(
            manifest_df=cdf,
            data_root=DATA_ROOT,
            window_size=DATASET_WINDOW,
            stride=DATASET_STRIDE
        )
        
        if len(dataset) == 0:
            print("  Skipping empty chunk.")
            continue

        # MODIFIED: Added tqdm progress bar (from Code 1)
        pbar = tqdm(range(len(dataset)), desc=f"  Chunk {ci+1}", ncols=100)
        
        for idx in pbar:
            # --- Define tensors outside try block for 'finally' ---
            clips = None
            features = None
            loader = None
            path = "Unknown"
            
            # MODIFIED: Added robust error handling (from Code 1)
            try:
                # --- 1. Load Clips ---
                clips, path = dataset[idx]
                if not clips:
                    print(f"Skipping short video: {path}")
                    continue
                
                # --- 2. Logging (from Code 1) ---
                clip_shape = clips[0].shape
                clips_ram_mb = (len(clips) * clips[0].nelement() * 4) / (1024**2)
                pbar.set_postfix_str(f"Video: {os.path.basename(path)}, Clips: {len(clips)}, RAM: {clips_ram_mb:.1f}MB")

                # --- 3. Extract Features ---
                features = extract_features_batch(clips, FEATURE_BATCH_SIZE)
                feat_shape = features.shape
                
                # --- 4. Logging (from Code 1) ---
                feat_ram_mb = (features.nelement() * 4) / (1024**2)
                pbar.set_postfix_str(f"Video: {os.path.basename(path)}, Feats: {feat_shape}, RAM: {feat_ram_mb:.1f}MB")
                
                # --- 5. Critical Cleanup (from Code 1) ---
                del clips 
                clips = None # Set to None
                
                # --- 6. Train Autoencoder ---
                loader = DataLoader(TensorDataset(features), batch_size=AE_BATCH_SIZE, shuffle=True)

                video_total_loss = 0.0
                for (bf,) in loader:
                    bf = bf.to(device)
                    recon, latent = autoencoder(bf)
                    loss_recon = criterion_recon(recon, bf)
                    loss_reg = latent_reg_weight * torch.mean(latent**2)
                    loss = loss_recon + loss_reg
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    video_total_loss += loss.item() * len(bf)
                    
                    # More granular cleanup
                    del bf, recon, latent, loss, loss_recon, loss_reg

                epoch_total_loss += video_total_loss
                epoch_total_samples += len(features)

            # MODIFIED: Specific Error Handling (from Code 1)
            except torch.cuda.OutOfMemoryError as e:
                print(f"\n  !! CUDA OOM ERROR on {os.path.basename(path)} !!")
                print(f"  Shape that failed (features): {feat_shape if features is not None else 'N/A'}")
                print(f"  Shape that failed (clips): {len(clips) if clips is not None else 'N/A'} x {clip_shape if clips is not None else 'N/A'}")
                print(f"  Skipping this video. Consider reducing FEATURE_BATCH_SIZE.")
            
            except Exception as e:
                print(f"\n  !! An unexpected error occurred on {os.path.basename(path)}: {e} !!")
                print("  Skipping this video.")
                
            # MODIFIED: Robust Cleanup 'finally' block (from Code 1)
            finally:
                # Explicitly delete all large tensors
                if features is not None:
                    del features
                if clips is not None:
                    del clips
                if loader is not None:
                    del loader
                    
                gc.collect()
                torch.cuda.empty_cache()
                
        # End of video loop
        del dataset, pbar # Clear dataset and pbar
        gc.collect()

    # --- End of Epoch Logic (from Code 1) ---
    if epoch_total_samples == 0:
        print(f"\nEpoch {epoch+1} completed, but no samples were processed. Check data paths.")
        continue

    avg_loss = epoch_total_loss / epoch_total_samples
    print(f"\n Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

    # --- 1. Save Latest Full Checkpoint (Improved) ---
    try:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, LATEST_CKPT_PATH)
        print(f"Latest checkpoint saved: {LATEST_CKPT_PATH}")
    except Exception as e:
        print(f"Error saving latest checkpoint: {e}")

    # --- 2. Early Stopping & Best Model Logic (from Code 1) ---
    if best_loss - avg_loss > min_delta:
        best_loss = avg_loss
        early_stop_counter = 0
        try:
            torch.save(autoencoder.state_dict(), BEST_MODEL_PATH)
            print(f"New best loss: {best_loss:.6f}. Best model updated!\n")
        except Exception as e:
            print(f"Error saving best model: {e}")
    else:
        early_stop_counter += 1
        print(f"No improvement. Early stopping counter: {early_stop_counter}/{patience}\n")
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break # Exit the main training loop

print("\nTraining completed successfully!")