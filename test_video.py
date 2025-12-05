import os
import gc
import torch # type: ignore
import torch.nn as nn # type: ignore
import torchvision.transforms as T # type: ignore
import torchvision.models.video as video_models # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
from tqdm import tqdm # type: ignore
from sklearn.metrics import confusion_matrix, classification_report # type: ignore

# =====================================================
# CONFIGURATION
# =====================================================
DATA_ROOT = r"C:\Users\Harshith\Downloads\intel_robotic_welding_dataset\raid\intel_robotic_welding_dataset"
MANIFEST_PATH = os.path.join(DATA_ROOT, "manifest.csv")
CHECKPOINT_DIR = os.path.join(DATA_ROOT, "checkpoints")
CHECKPOINT_VAL_DIR = os.path.join(DATA_ROOT, "checkpoints_val")

LATEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
CLASSIFIER_PATH = os.path.join(CHECKPOINT_VAL_DIR, "val_latest_checkpoint.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================
# DATASET CLASS
# =====================================================
class WeldVideoManifestDataset:
    def __init__(self, manifest_df, data_root, window_size=16, stride=8):
        self.video_paths, self.labels = [], []
        self.categories = sorted(manifest_df["CATEGORY"].unique())
        self.label_map = {cat: i for i, cat in enumerate(self.categories)}

        for _, row in manifest_df.iterrows():
            subdir_path = os.path.join(data_root, row["SUBDIRS"])
            subdir_name = os.path.basename(row["SUBDIRS"])
            video_file = os.path.join(subdir_path, f"{subdir_name}.avi")
            if os.path.exists(video_file):
                self.video_paths.append(video_file)
                self.labels.append(self.label_map[row["CATEGORY"]])

        self.window_size = window_size
        self.stride = stride
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.video_paths)

    def get_video_clips(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame_rgb))
        cap.release()

        if len(frames) < self.window_size:
            return []
        clips = []
        for i in range(0, len(frames) - self.window_size + 1, self.stride):
            clips.append(torch.stack(frames[i:i+self.window_size], dim=1))
        return clips

# =====================================================
# MODEL DEFINITIONS
# =====================================================
feature_net = video_models.r3d_18(weights="KINETICS400_V1").to(device)
feature_net.eval()
print("Feature extractor (r3d_18) loaded.")

# FIX 1: Correct AutoEncoder default dimension to match training
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

# Load AutoEncoder
autoencoder = AutoEncoder(input_dim=400, bottleneck_dim=64).to(device)
checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
if "model_state_dict" in checkpoint:
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded AutoEncoder from checkpoint (epoch {checkpoint.get('epoch', '?')})")
else:
    autoencoder.load_state_dict(checkpoint)
    print("Loaded AutoEncoder from checkpoint (raw state dict)")
autoencoder.eval()

# Classifier
class WeldClassifier(nn.Module):
    def __init__(self, bottleneck_dim=64, num_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# =====================================================
# FEATURE EXTRACTION HELPERS (FIXED)
# =====================================================
@torch.no_grad()
def extract_features(video_clips):
    """
    Extract features from video clips using r3d_18.
    FIX: Removed incorrect .permute() that was causing dimension mismatch.
    r3d_18 expects input: [batch, channels, time, height, width]
    Our clips are already in format: [channels=3, time=16, height=112, width=112]
    """
    feats = []
    for i in range(0, len(video_clips), 8):
        batch = torch.stack(video_clips[i:i+8]).to(device)  # [batch, 3, 16, 112, 112]
        f = feature_net(batch)  # FIX: No permute needed! Outputs [batch, 512]
        f = f.view(f.size(0), -1)  # Ensure flattened: [batch, 512]
        feats.append(f.cpu())
        del batch, f
        torch.cuda.empty_cache()
    
    if not feats:
        return torch.empty(0, 512)  # Return correct dimension
    
    all_features = torch.cat(feats, dim=0)  # [total_clips, 512]
    return all_features

@torch.no_grad()
def get_latent_representation(video_clips):
    """
    Get latent representation from video clips.
    FIX: Handle dimension mismatch between r3d_18 output (512) and AutoEncoder input (400)
    """
    features = extract_features(video_clips)  # [N, 512]
    
    if features.numel() == 0:
        return None
    
    # FIX 2: Handle 512->400 dimension reduction
    # CRITICAL: Your training code should have this dimension reduction!
    # For now, we truncate to 400 dimensions (not ideal, but matches your checkpoint)
    # BETTER SOLUTION: Retrain with input_dim=512 OR add a learned projection layer
    features_reduced = features[:, :400]  # Truncate from 512 to 400
    
    # Pass through autoencoder to get latent representation
    _, latent = autoencoder(features_reduced.to(device))
    
    # Return mean latent across all clips for this video
    return latent.mean(dim=0)

# =====================================================
# LOAD TEST SPLIT
# =====================================================
df = pd.read_csv(MANIFEST_PATH)
df.rename(columns=lambda x: x.strip().upper(), inplace=True)

if "CATEGORY" not in df.columns or "SPLIT" not in df.columns:
    raise ValueError("manifest.csv must have CATEGORY and SPLIT columns.")

test_df = df[df["SPLIT"] == "TEST"].reset_index(drop=True)
dataset = WeldVideoManifestDataset(test_df, DATA_ROOT)

print(f"\nLoaded {len(dataset)} TEST videos with {len(dataset.categories)} categories.")
print("Categories:", dataset.categories)

# =====================================================
# LOAD CLASSIFIER (FIXED)
# =====================================================
clf = WeldClassifier(bottleneck_dim=64, num_classes=len(dataset.categories)).to(device)

# FIX 3: Properly load classifier checkpoint
clf_checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
if "model_state_dict" in clf_checkpoint:
    clf.load_state_dict(clf_checkpoint["model_state_dict"])
    print(f"Loaded classifier from checkpoint (epoch {clf_checkpoint.get('epoch', '?')}, acc: {clf_checkpoint.get('acc', '?'):.2%})")
else:
    clf.load_state_dict(clf_checkpoint)
    print("Loaded classifier from checkpoint (raw state dict)")
clf.eval()

# =====================================================
# PREDICTION & EVALUATION
# =====================================================
print("\n" + "="*60)
print("Starting evaluation on TEST set...")
print("="*60)

true_labels, pred_labels, video_names = [], [], []
skipped_videos = []

for i in tqdm(range(len(dataset)), desc="Evaluating videos"):
    video_path = dataset.video_paths[i]
    label = dataset.labels[i]
    
    try:
        clips = dataset.get_video_clips(video_path)
        if not clips:
            skipped_videos.append((os.path.basename(video_path), "Too short"))
            continue

        latent = get_latent_representation(clips)
        if latent is None:
            skipped_videos.append((os.path.basename(video_path), "No features"))
            continue

        pred = clf(latent.unsqueeze(0).to(device))
        pred_class = pred.argmax(dim=1).item()

        true_labels.append(label)
        pred_labels.append(pred_class)
        video_names.append(os.path.basename(video_path))

    except Exception as e:
        skipped_videos.append((os.path.basename(video_path), f"Error: {str(e)[:50]}"))
        print(f"\n  Error processing {os.path.basename(video_path)}: {e}")
    
    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

# =====================================================
# SUMMARY METRICS
# =====================================================
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

if skipped_videos:
    print(f"\n Skipped {len(skipped_videos)} videos:")
    for vid, reason in skipped_videos[:10]:  # Show first 10
        print(f"   - {vid}: {reason}")
    if len(skipped_videos) > 10:
        print(f"   ... and {len(skipped_videos) - 10} more")

true_np, pred_np = np.array(true_labels), np.array(pred_labels)
acc = (true_np == pred_np).mean() * 100 if len(true_np) > 0 else 0

print(f"\n{'='*60}")
print(f"Overall Accuracy: {acc:.2f}% on {len(true_np)} test videos")
print(f"{'='*60}")

print("\nConfusion Matrix:")
cm = confusion_matrix(true_np, pred_np)
print(cm)

print("\n" + "="*60)
print("Detailed Classification Report:")
print("="*60)
report = classification_report(true_np, pred_np, target_names=dataset.categories, zero_division=0)
print(report)

# =====================================================
# SAVE RESULTS
# =====================================================
results_df = pd.DataFrame({
    "Video": video_names,
    "True_Label": [dataset.categories[t] for t in true_labels],
    "Predicted_Label": [dataset.categories[p] for p in pred_labels],
    "Correct": [true_labels[i] == pred_labels[i] for i in range(len(true_labels))]
})

results_csv = os.path.join(DATA_ROOT, "test_predictions.csv")
results_df.to_csv(results_csv, index=False)
print(f"\n Saved predictions to: {results_csv}")

# Save confusion matrix
cm_df = pd.DataFrame(cm, index=dataset.categories, columns=dataset.categories)
cm_csv = os.path.join(DATA_ROOT, "confusion_matrix.csv")
cm_df.to_csv(cm_csv)
print(f" Saved confusion matrix to: {cm_csv}")

print("\n" + "="*60)
print("Evaluation completed successfully!")
print("="*60)