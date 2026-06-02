import os
import sys
from pathlib import Path
import argparse
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from tqdm import tqdm

# ---------------------------
# Config (user-tweakable)
# ---------------------------
DINOV2_REPO = "./dinov2"
PRETRAINED_WEIGHTS = "./weights/dinov2_vitb14_reg4_pretrain.pth"
DATA_ROOT = "./dataset"
SAVE_DIR = "./outputs"
TEXT_FEAT_PATH = "./text_features/genus_text_features.pt"

NUM_CLASSES = 44
IMAGE_SIZE = 518
BATCH_SIZE = 32
NUM_WORKERS = min(8, os.cpu_count())
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LORA_R = 4
LORA_ALPHA = 16

LAMBDA_ALIGN = 0.1

# ---------------------------
# LoRA modules
# ---------------------------
class LoRALinear(nn.Module):
    def __init__(self, orig: nn.Linear, r: int = 4, alpha: float = 16.0):
        super().__init__()
        self.orig = orig
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
            self.B = nn.Parameter(torch.zeros(self.out_features, r))
            self.scaling = alpha / max(1, r)
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.orig(x)
        if self.r > 0:
            BA = torch.matmul(self.B, self.A) * (self.scaling)
            delta = F.linear(x, BA)
            out = out + delta
        return out


class QKVLoRA(nn.Module):
    def __init__(self, orig: nn.Linear, r: int = 4, alpha: float = 16.0):
        super().__init__()
        self.orig = orig
        assert orig.out_features % 3 == 0, 'expected qkv linear with out_features divisible by 3'
        self.embed_dim = orig.out_features // 3
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A_q = nn.Parameter(torch.randn(r, orig.in_features) * 0.01)
            self.B_q = nn.Parameter(torch.zeros(self.embed_dim, r))
            self.A_v = nn.Parameter(torch.randn(r, orig.in_features) * 0.01)
            self.B_v = nn.Parameter(torch.zeros(self.embed_dim, r))
            self.scaling = alpha / max(1, r)
        else:
            self.register_parameter('A_q', None)
            self.register_parameter('B_q', None)
            self.register_parameter('A_v', None)
            self.register_parameter('B_v', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.orig(x)
        if self.r > 0:
            q, k, v = out.chunk(3, dim=-1)
            BA_q = torch.matmul(self.B_q, self.A_q) * (self.scaling)
            BA_v = torch.matmul(self.B_v, self.A_v) * (self.scaling)
            delta_q = F.linear(x, BA_q)
            delta_v = F.linear(x, BA_v)
            q = q + delta_q
            v = v + delta_v
            out = torch.cat([q, k, v], dim=-1)
        return out


# ---------------------------
# Projection heads
# ---------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TextProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_cls_from_backbone(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(backbone, 'forward_features'):
        out = backbone.forward_features(x)
    else:
        out = backbone(x)

    if isinstance(out, dict):
        if 'x_norm_clstoken' in out:
            return out['x_norm_clstoken']
        if 'cls' in out:
            return out['cls']
        if 'last_hidden_state' in out:
            return out['last_hidden_state'][:, 0]
        for v in out.values():
            if isinstance(v, torch.Tensor) and v.dim() == 3:
                return v[:, 0]
        raise RuntimeError('Backbone dict output: no CLS-like tensor found')
    if isinstance(out, torch.Tensor):
        if out.dim() == 3:
            return out[:, 0]
        if out.dim() == 2:
            return out
    raise RuntimeError('Unexpected backbone output type/shape')


def supcon_loss_from_batch(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    device = z.device
    z = F.normalize(z, dim=1)
    N = z.size(0)

    sim = torch.matmul(z, z.t()) / temperature
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.t()).to(device)
    diag = torch.eye(N, dtype=torch.bool, device=device)
    mask_pos = mask_pos & (~diag)

    sim_masked = sim.masked_fill(diag, -1e9)
    lse = torch.logsumexp(sim_masked, dim=1)
    pos_counts = mask_pos.sum(dim=1)
    sum_pos_sim = (sim * mask_pos.float()).sum(dim=1)
    mean_pos_sim = sum_pos_sim / pos_counts.clamp(min=1).float()
    loss_per = lse - mean_pos_sim

    valid = pos_counts > 0
    if valid.sum() == 0:
        return (z.sum() * 0.0)
    loss = loss_per[valid].mean()
    return loss


def asymmetric_align_loss(img_proj, txt_proj, labels):

    unique_labels = torch.unique(labels)
    loss_align = 0.0
    

    img_proj = F.normalize(img_proj, dim=1)
    txt_proj = F.normalize(txt_proj, dim=1)

    for label in unique_labels:
        mask = (labels == label)

        img_prototype = img_proj[mask].mean(dim=0).detach()
        img_prototype = F.normalize(img_prototype, dim=0)

        txt_feature = txt_proj[mask][0]

        similarity = torch.dot(img_prototype, txt_feature)
        loss_align += (1.0 - similarity)
        
    return loss_align / len(unique_labels)

def inject_lora(model: nn.Module, r: int = 4, alpha: int = 16):
    replaced = []
    for i, blk in enumerate(model.blocks):
        orig_qkv = blk.attn.qkv
        blk.attn.qkv = QKVLoRA(orig_qkv, r=r, alpha=alpha)
        replaced.append(f'blocks.{i}.attn.qkv')
    print(f'Injected LoRA into {len(replaced)} attention qkv layers')
    return model

def build_backbone(dinov2_repo: str, pretrained_weights: str) -> nn.Module:
    sys.path.insert(0, dinov2_repo)
    try:
        from dinov2.models.vision_transformer import vit_base
    except Exception as e:
        raise RuntimeError(f"Import error: {e}")

    model = vit_base(
        patch_size=14,
        img_size=IMAGE_SIZE,
        init_values=1.0,
        block_chunks=0,
        num_register_tokens=4,
    )

    if pretrained_weights and os.path.exists(pretrained_weights):
        ckpt = torch.load(pretrained_weights, map_location='cpu')
        if 'model' in ckpt:
            ckpt = ckpt['model']
        model.load_state_dict(ckpt, strict=False)
    else:
        raise RuntimeError(f"Pretrained weights not found: {pretrained_weights}")

    return model


def infer_embed_dim_from_backbone(backbone: nn.Module) -> int:
    if hasattr(backbone, 'embed_dim'):
        val = getattr(backbone, 'embed_dim')
        if isinstance(val, int) and val > 0:
            return val
    pe = getattr(backbone, 'patch_embed', None)
    if pe is not None:
        proj = getattr(pe, 'proj', None)
        if proj is not None and hasattr(proj, 'out_channels'):
            val = getattr(proj, 'out_channels')
            if isinstance(val, int) and val > 0:
                return val
    for module in backbone.modules():
        if isinstance(module, nn.Linear) and module.out_features % 3 == 0:
            return module.out_features // 3
    raise RuntimeError('Could not infer embed_dim from backbone')

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes_per_batch, n_samples_per_class):
        self.labels = list(labels)
        self.label2indices = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.label2indices[lbl].append(idx)
        self.classes = list(self.label2indices.keys())
        self.n_classes = n_classes_per_batch
        self.n_samples = n_samples_per_class
        self.batch_size = self.n_classes * self.n_samples
        self.num_samples = len(self.labels)
        self.num_batches = max(1, self.num_samples // self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            selected_classes = random.sample(self.classes, self.n_classes)
            batch = []
            for c in selected_classes:
                idxs = self.label2indices[c]
                if len(idxs) >= self.n_samples:
                    chosen = random.sample(idxs, self.n_samples)
                else:
                    chosen = random.choices(idxs, k=self.n_samples)
                batch.extend(chosen)
            random.shuffle(batch)
            yield batch

def get_model_state_dict(backbone: nn.Module, proj: nn.Module, proj_txt: nn.Module):
    sd = backbone.state_dict()
    keep_backbone = {}
    for k, v in sd.items():
        if (
            k.endswith('.A_q') or k.endswith('.B_q') or
            k.endswith('.A_v') or k.endswith('.B_v')
        ):
            keep_backbone[k] = v
    
    return {
        'backbone_lora': keep_backbone, 
        'proj_img': proj.state_dict(),
        'proj_txt': proj_txt.state_dict() 
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DATA_ROOT)
    parser.add_argument('--weights', default=PRETRAINED_WEIGHTS)
    parser.add_argument('--dinov2_repo', default=DINOV2_REPO)
    parser.add_argument('--save_dir', default=SAVE_DIR)
    parser.add_argument('--text_feat_path', default=TEXT_FEAT_PATH) 
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE)
    parser.add_argument('--r', type=int, default=LORA_R)
    parser.add_argument('--alpha', type=int, default=LORA_ALPHA)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--lambda_align', type=float, default=LAMBDA_ALIGN) 
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / 'train.log'

    if not log_file.exists():
        with open(log_file, 'w') as f:
            f.write('epoch,train_loss,train_loss_sup,train_loss_align,val_supcon_loss\n')

    print('Loading text features...')
    if os.path.exists(args.text_feat_path):
        text_features_bank = torch.load(args.text_feat_path).to(DEVICE)
        text_features_bank.requires_grad = False
        txt_dim = text_features_bank.shape[1]
        print(f"Loaded text features: {text_features_bank.shape}")
    else:
        raise RuntimeError(f"Text feature file not found at {args.text_feat_path}")

    print('Building backbone...')
    backbone = build_backbone(args.dinov2_repo, args.weights)
    embed_dim = infer_embed_dim_from_backbone(backbone)
    print(f'Inferred embed_dim = {embed_dim}')

    for p in backbone.parameters():
        p.requires_grad = False
    print('Backbone parameters frozen')
    
    inject_lora(backbone, r=args.r, alpha=args.alpha)
    backbone.to(DEVICE)

    proj = ProjectionHead(embed_dim, hidden_dim=256, out_dim=128).to(DEVICE)
    proj_txt = TextProjectionHead(in_dim=txt_dim, hidden_dim=512, out_dim=128).to(DEVICE)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.14)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    train_ds = ImageFolder(train_dir, transform=train_transforms)
    val_ds = ImageFolder(val_dir, transform=val_transforms)

    n_samples_per_class = 2
    n_classes_per_batch = args.batch_size // n_samples_per_class
    train_sampler = BalancedBatchSampler(train_ds.targets, n_classes_per_batch=n_classes_per_batch,
                                         n_samples_per_class=n_samples_per_class)
    val_sampler = BalancedBatchSampler(val_ds.targets, n_classes_per_batch=n_classes_per_batch,
                                       n_samples_per_class=n_samples_per_class)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True)

    trainable = [p for p in list(backbone.parameters()) + list(proj.parameters()) + list(proj_txt.parameters()) if p.requires_grad]
    print(f'Trainable parameters: {sum(p.numel() for p in trainable)}')
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_val_supcon = float('inf')

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        proj.train()
        proj_txt.train() 

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]')
        running_loss = 0.0
        running_loss_sup = 0.0
        running_loss_align = 0.0
        n_samples = 0

        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            cls = get_cls_from_backbone(backbone, images)
            z_img = proj(cls)

            batch_txt_feats = text_features_bank[labels] # [Batch, 768]
            z_txt = proj_txt(batch_txt_feats) # [Batch, 128]

            loss_sup = supcon_loss_from_batch(z_img, labels, temperature=0.07)
            loss_align = asymmetric_align_loss(z_img, z_txt, labels)
            loss = loss_sup + args.lambda_align * loss_align

            loss.backward()
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_loss_sup += loss_sup.item() * bs
            running_loss_align += loss_align.item() * bs
            n_samples += bs
            pbar.set_postfix({
                'L_tot': loss.item(), 
                'L_sup': loss_sup.item(),
                'L_aln': loss_align.item()
            })

        epoch_loss = running_loss / max(1, n_samples)
        epoch_loss_sup = running_loss_sup / max(1, n_samples)
        epoch_loss_align = running_loss_align / max(1, n_samples)

        backbone.eval()
        proj.eval()
        total_val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                cls = get_cls_from_backbone(backbone, images)
                z = proj(cls)
                batch_loss = supcon_loss_from_batch(z, labels, temperature=0.07)
                total_val_loss += batch_loss.item()
                n_batches += 1
        val_supcon_loss = total_val_loss / n_batches if n_batches > 0 else float('inf')

        print(f'Epoch {epoch} | Total: {epoch_loss:.4f} | Sup: {epoch_loss_sup:.4f} | Align: {epoch_loss_align:.4f} | ValSup: {val_supcon_loss:.6f}')
        
        with open(log_file, 'a') as f:
            f.write(f'{epoch},{epoch_loss:.6f},{epoch_loss_sup:.6f},{epoch_loss_align:.6f},{val_supcon_loss:.6f}\n')

        state = {
            'epoch': epoch,
            'model_state_dict': get_model_state_dict(backbone, proj, proj_txt),
            'val_supcon_loss': val_supcon_loss,
            'args': vars(args)
        }

        if epoch % 10 == 0:
            torch.save(state, save_dir / f'checkpoint_epoch{epoch}.pth')
            print(f'Saved checkpoint at epoch {epoch}')

        if val_supcon_loss < best_val_supcon:
            best_val_supcon = val_supcon_loss
            torch.save(state, save_dir / 'best_lora_proj.pth')
            print('Saved best LoRA + proj checkpoint')

    print('Training finished.')
    print(f'Best supcon_loss = {best_val_supcon:.4f}')


if __name__ == '__main__':
    main()
