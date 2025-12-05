import os
import gc
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models.video as video_models
import cv2
import pandas as pd

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
# DATASET UTILS (minimal version)
# =====================================================
class WeldVideoReader:
    def __init__(self, window_size=16, stride=8):
        self.window_size = window_size
        self.stride = stride
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor()
        ])

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
ae_checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
autoencoder.load_state_dict(ae_checkpoint["model_state_dict"] if "model_state_dict" in ae_checkpoint else ae_checkpoint)
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

# Load Classifier
clf = WeldClassifier(bottleneck_dim=64).to(device)
clf_checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
clf.load_state_dict(clf_checkpoint["model_state_dict"] if "model_state_dict" in clf_checkpoint else clf_checkpoint)
clf.eval()

# =====================================================
# FEATURE EXTRACTION HELPERS
# =====================================================
@torch.no_grad()
def extract_features(video_clips):
    feats = []
    for i in range(0, len(video_clips), 8):
        batch = torch.stack(video_clips[i:i+8]).to(device)  # [batch, 3, 16, 112, 112]
        f = feature_net(batch)
        f = f.view(f.size(0), -1)
        feats.append(f.cpu())
        del batch, f
        torch.cuda.empty_cache()
    if not feats:
        return torch.empty(0, 512)
    return torch.cat(feats, dim=0)

@torch.no_grad()
def get_latent_representation(video_clips):
    features = extract_features(video_clips)
    if features.numel() == 0:
        return None
    features_reduced = features[:, :400]  # truncate 512→400
    _, latent = autoencoder(features_reduced.to(device))
    return latent.mean(dim=0)

# =====================================================
# MAIN PREDICTION FUNCTION
# =====================================================
def predict_weld_type(video_path):
    # Load categories from manifest
    df = pd.read_csv(MANIFEST_PATH)
    df.rename(columns=lambda x: x.strip().upper(), inplace=True)
    categories = sorted(df["CATEGORY"].unique())

    reader = WeldVideoReader()
    clips = reader.get_video_clips(video_path)

    if not clips:
        print(" Video too short or unreadable.")
        return None

    latent = get_latent_representation(clips)
    if latent is None:
        print(" No valid features extracted.")
        return None

    pred = clf(latent.unsqueeze(0).to(device))
    pred_class = pred.argmax(dim=1).item()
    print(f"\n Predicted weld type: {categories[pred_class]}")
    return categories[pred_class]

# =====================================================
# EXAMPLE USAGE
# =====================================================
if __name__ == "__main__":
    video_path = r"C:\Users\Harshith\Downloads\intel_robotic_welding_dataset\raid\intel_robotic_welding_dataset\porosity_w-excessive_penetration_10_11_01_22_butt_joint\11-01-22-0172-04\11-01-22-0172-04.avi"  # <-- change this
    predict_weld_type(video_path)
