WeldNet: Real-Time Unsupervised Welding Defect Classification


A production-ready PyTorch pipeline for classifying 12 robotic welding conditions directly from RGB video using the Intel Robotic Welding Dataset (2024).

WeldNet combines unsupervised latent learning + lightweight supervised classification, inspired by modern SSL representation learning (e.g., arXiv:2409.02290) but fully optimized for industrial robotic welding.

Highlights

No temporal or frame-level labels required

R3D-18 (Kinetics-400 pretrained) + Autoencoder bottleneck (64-d)

Memory-efficient training with video chunking

Complete training → validation → evaluation pipeline (5 commands)

Project Structure
intel_robotic_welding_dataset/
├── raid/
│   └── intel_robotic_welding_dataset/
│       ├── manifest.csv
│       ├── good_weld_.../
│       ├── lack_of_fusion_.../
│       └── ...
├── checkpoints/                  # Autoencoder weights
│   ├── latest_checkpoint.pt
│   └── best_model.pt
├── checkpoints_val/              # Classifier weights
│   ├── val_latest_checkpoint.pt
│   └── val_best_model.pt
├── train_welding_model.py        # Stage 1: Unsupervised AE training
├── val_welding_model.py          # Stage 2: Classifier training
├── test_video.py                 # TEST split evaluation → CSV + confusion matrix
├── single_video.py               # Real-time inference on any .avi file
├── check_file.py                 # Inspect .pt checkpoints
└── README.md

 Quick Start (5 Commands)
1. Train the Autoencoder (Unsupervised Stage)
python train_welding_model.py

2. Train the Classifier on Frozen Latents
python val_welding_model.py

3. Evaluate on the Official TEST Split

Generates:

confusion matrix

per-class accuracy

predictions CSV

python test_video.py

4. Real-Time Inference on Any Video

Edit video_path inside the script:

python single_video.py

5. Inspect a Checkpoint
python check_file.py

 Model Pipeline Overview
Video (.avi)
   ↓ (Sliding window: 16 frames, stride 8–128)
Clip batch
   ↓
R3D-18 (Kinetics-400 pretrained)
   ↓   → 512-d features
Truncate to 400 dims
   ↓
Autoencoder (400 → 64 → 400)
   ↓   → mean-pooled 64-d latent
2-layer MLP classifier
   ↓
12-class prediction

🛠️ Technical Details

Backbone:

torchvision.models.video.r3d_18(weights="KINETICS400_V1")

Autoencoder:

Architecture: 400 → 256 → 64 → 256 → 400

Loss: SmoothL1 + latent L2 (1e-4)

Classifier:

MLP: 64 → 128 → 12

Loss: Cross-Entropy

Optimizer: Adam

Environment:

PyTorch 2.3+

CUDA 11.8 / 12.x

OpenCV 4.8+
 References

Intel Corporation, Intel Robotic Welding Dataset, 2024.

Repo credits : https://huggingface.co/datasets/IntelLabs/Intel_Robotic_Welding_Multimodal_Dataset
W. Kay et al., “The Kinetics Human Action Video Dataset,” arXiv:1705.06950 (2017).

K. Hara et al., “Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs?,” CVPR 2018.

T. Chen et al., “Self-Supervised Learning of Visual Representations from Uncurated Data,” arXiv:2409.02290 (2024).
