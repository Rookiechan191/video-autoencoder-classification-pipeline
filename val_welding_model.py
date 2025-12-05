import os
import gc
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.models.video as video_models
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================
DATA_ROOT = r"C:\Users\Harshith\Downloads\intel_robotic_welding_dataset\raid\intel_robotic_welding_dataset"
MANIFEST_PATH = os.path.join(DATA_ROOT, "manifest.csv")

CHECKPOINT_DIR = os.path.join(DATA_ROOT, "checkpoints")
CHECKPOINT_VAL_DIR = os.path.join(DATA_ROOT, "checkpoints_val")
os.makedirs(CHECKPOINT_VAL_DIR, exist_ok=True)

LATEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================
# DATASET CLASS
# =====================================================
class WeldVideoManifestDataset(Dataset):
    def __init__(self, manifest_df, data_root, window_size=16, stride=128):
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

    def __getitem__(self, idx):
        clips = self.get_video_clips(self.video_paths[idx])
        label = self.labels[idx]
        return clips, label

# =====================================================
# MODEL DEFINITIONS
# =====================================================
feature_net = video_models.r3d_18(weights="KINETICS400_V1").to(device)
feature_net.eval()

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

#  FIX: Load latest autoencoder checkpoint properly
checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
if "model_state_dict" in checkpoint:
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model weights from checkpoint (epoch {checkpoint.get('epoch', '?')})")
else:
    autoencoder.load_state_dict(checkpoint)
    print("Loaded raw state_dict (no wrapper detected).")
autoencoder.eval()

# =====================================================
# CLASSIFIER DEFINITION
# =====================================================
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
# FEATURE EXTRACTION
# =====================================================
@torch.no_grad()
def extract_features(video_clips):
    feats = []
    for i in range(0, len(video_clips), 8):
        batch = torch.stack(video_clips[i:i+8]).to(device)
        f = feature_net(batch)
        feats.append(f.cpu())
    if not feats:
        return torch.empty(0, 400)
    return torch.cat(feats, dim=0)

@torch.no_grad()
def get_latent_representation(video_clips):
    features = extract_features(video_clips)
    _, latent = autoencoder(features.to(device))
    return latent.mean(dim=0)

# =====================================================
# LOAD VALIDATION SPLIT
# =====================================================
df = pd.read_csv(MANIFEST_PATH)
val_df = df[df["SPLIT"] == "VAL"].reset_index(drop=True)
dataset = WeldVideoManifestDataset(val_df, DATA_ROOT)
print(f"Loaded {len(dataset)} validation videos.")

latents, labels = [], []
for i in tqdm(range(len(dataset)), desc="Extracting features"):
    clips, label = dataset[i]
    if not clips:
        continue
    latent = get_latent_representation(clips)
    latents.append(latent.cpu().numpy())
    labels.append(label)
    torch.cuda.empty_cache()
    gc.collect()

X = torch.tensor(np.vstack(latents), dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# =====================================================
# TRAIN CLASSIFIER
# =====================================================
num_classes = len(dataset.categories)
clf = WeldClassifier(bottleneck_dim=64, num_classes=num_classes).to(device)
optimizer = optim.Adam(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print(f"Training classifier on {len(X)} samples across {num_classes} classes.")

start_epoch = 0
resume_path = os.path.join(CHECKPOINT_VAL_DIR, "val_latest_checkpoint.pt")
if os.path.exists(resume_path):
    chkpt = torch.load(resume_path, map_location=device)
    clf.load_state_dict(chkpt["model_state_dict"])
    optimizer.load_state_dict(chkpt["optimizer_state_dict"])
    start_epoch = chkpt["epoch"] + 1
    print(f"Resumed training from epoch {start_epoch}")

num_epochs = 20
best_acc = 0.0

for epoch in range(start_epoch, num_epochs):
    clf.train()
    total_loss, correct = 0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        preds = clf(bx)
        loss = criterion(preds, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(bx)
        correct += (preds.argmax(1) == by).sum().item()

    avg_loss = total_loss / len(X)
    acc = correct / len(X)
    print(f"Epoch {epoch+1:02d}: Loss={avg_loss:.4f} | Acc={acc*100:.2f}%")

    # Save every epoch
    save_path = os.path.join(CHECKPOINT_VAL_DIR, f"val_epoch_{epoch+1}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": clf.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "acc": acc
    }, save_path)

    # Save latest checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": clf.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "acc": acc
    }, resume_path)

    # Track best model
    if acc > best_acc:
        best_acc = acc
        torch.save(clf.state_dict(), os.path.join(CHECKPOINT_VAL_DIR, "val_best_model.pt"))

print("\n Classifier training complete and all checkpoints saved!")
