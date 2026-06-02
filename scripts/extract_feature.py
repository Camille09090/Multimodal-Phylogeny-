import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
DINOV2_REPO = "./dinov2"

DATA_ROOT = "./dataset"
PRETRAINED_WEIGHTS = "./weights/dinov2_vitb14_reg4_pretrain.pth"
LORA_CHECKPOINT = "./checkpoints/best_lora_proj.pth"
OUTPUT_DIR = "./features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch.nn as nn
import torch.nn.functional as F

class QKVLoRA(nn.Module):
    def __init__(self, orig: nn.Linear, r=4, alpha=16):
        super().__init__()
        self.orig = orig
        self.embed_dim = orig.out_features // 3
        self.scaling = alpha / r

        self.A_q = nn.Parameter(torch.zeros(r, orig.in_features))
        self.B_q = nn.Parameter(torch.zeros(self.embed_dim, r))
        self.A_v = nn.Parameter(torch.zeros(r, orig.in_features))
        self.B_v = nn.Parameter(torch.zeros(self.embed_dim, r))

    def forward(self, x):
        out = self.orig(x)
        q, k, v = out.chunk(3, dim=-1)

        dq = F.linear(x, self.B_q @ self.A_q * self.scaling)
        dv = F.linear(x, self.B_v @ self.A_v * self.scaling)

        q = q + dq
        v = v + dv
        return torch.cat([q, k, v], dim=-1)


def inject_lora(model, r=4, alpha=16):
    for blk in model.blocks:
        blk.attn.qkv = QKVLoRA(blk.attn.qkv, r=r, alpha=alpha)
    return model


sys.path.insert(0, DINOV2_REPO)
from dinov2.models.vision_transformer import vit_base

model = vit_base(
    patch_size=14,
    img_size=518,
    init_values=1.0,
    block_chunks=0,
    num_register_tokens=4,
)

state = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
if "model" in state:
    state = state["model"]
model.load_state_dict(state, strict=False)

inject_lora(model, r=4, alpha=16)

ckpt = torch.load(LORA_CHECKPOINT, map_location="cpu")
lora_backbone_state = ckpt["model_state_dict"]["backbone_lora"]

model.load_state_dict(lora_backbone_state, strict=False)
with torch.no_grad():
    total = 0.0
    count = 0
    for name, p in model.named_parameters():
        if any(k in name for k in ["A_q", "B_q", "A_v", "B_v"]):
            total += p.abs().mean().item()
            count += 1

    print("LoRA param tensors:", count)
    print("Mean |LoRA params|:", total / max(1, count))


model.eval()
model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
])

splits = ["train", "val", "test"]
class_names = sorted({
    cls
    for split in splits
    for cls in os.listdir(os.path.join(DATA_ROOT, split))
    if os.path.isdir(os.path.join(DATA_ROOT, split, cls))
})

label_to_index = {cls: i for i, cls in enumerate(class_names)}
index_to_label = {i: cls for cls, i in label_to_index.items()}
features = []
labels = []

with torch.no_grad():
    for split in splits:
        print(f"\nProcessing split: {split}")
        split_dir = os.path.join(DATA_ROOT, split)

        for cls in class_names:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                continue

            for img_name in tqdm(os.listdir(cls_dir), desc=f"{split}/{cls}"):
                if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(DEVICE)

                out = model.forward_features(img)
                cls_feat = out["x_norm_clstoken"]  # [1, 768]

                features.append(cls_feat.squeeze(0).cpu().numpy())
                labels.append(label_to_index[cls])

features = np.asarray(features, dtype=np.float32)
labels = np.asarray(labels, dtype=np.int64)

np.save(os.path.join(OUTPUT_DIR, "features.npy"), features)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
np.save(os.path.join(OUTPUT_DIR, "index_to_label.npy"), index_to_label)

print("\n===================================")
print(" SupCon + LoRA CLS feature extraction done")
print("features shape:", features.shape)  # (N, 768)
print("labels shape:", labels.shape)
print("outdir:", OUTPUT_DIR)
