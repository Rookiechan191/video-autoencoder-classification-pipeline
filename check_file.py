import torch
import os

# 🔁 Update this path to your actual checkpoint file
ckpt_path = r"C:\Users\Harshith\Downloads\intel_robotic_welding_dataset\raid\intel_robotic_welding_dataset\checkpoints\latest_checkpoint.pt"

# ✅ Check if file exists
if not os.path.exists(ckpt_path):
    print(f"❌ File not found: {ckpt_path}")
else:
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt, dict):
        print("✅ It's a dictionary checkpoint.")
        print("Keys inside:", ckpt.keys())

        # 👀 If it looks like a state_dict, show a preview
        if 'model_state_dict' in ckpt and 'optimizer_state_dict' in ckpt:
            print("\n✅ Detected full checkpoint structure.")
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            print("\n⚠️ Looks like a pure model state_dict (weights only).")
        else:
            print("\nℹ️ Contains non-tensor data; inspect manually.")
    else:
        print(f"⚠️ It's a raw object of type: {type(ckpt)}")
