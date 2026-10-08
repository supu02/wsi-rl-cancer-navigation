"""
SASHA Step 2 - UNI2 Feature Extraction

This script replaces SASHA's Step 2 feature extraction with UNI2-h embeddings.
UNI2-h is a ViT-Giant/14 (1536-dim class token) loaded via timm from MahmoodLab/UNI2-h.

Key differences from Virchow variant:
- Uses UNI2-h instead of Virchow2
- No SwiGLUPacked required; standard timm load
- Raw 1536-dim class token saved directly — NO random projection
  (Virchow variant used a random 2560->384 DimensionReducer; we do not replicate that here.
   Downstream configs must use D_feat: 1536 for UNI2 runs.)
- Outputs are isolated under features_uni2_subset16 (never touches Virchow outputs)

For CAMELYON16 smoke test (one slide, 8 patches max):
python sasha_adapted/preprocessing/step2_extract_uni2_features.py \
    --dataset_name camelyon16 \
    --data_h5_dir /usr/users/rl_for_wsi/malla_sup/projects/SASHA/outputs/patches_subset16 \
    --data_slide_dir /usr/users/rl_for_wsi/malla_sup/projects/SASHA/subset16_slides \
    --slide_ext .tif \
    --csv_path /usr/users/rl_for_wsi/malla_sup/projects/SASHA/dataset_csv/camelyon16_subset16/camelyon16_subset16.csv \
    --feat_dir outputs/features_uni2_subset16 \
    --batch_size 4 \
    --extract_high_res_features True \
    --patch_level_low_res 3 \
    --patch_level_high_res 1 \
    --smoke_test \
    --max_slides 1 \
    --max_patches_per_slide 8
"""

import argparse
import os
import time

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm
import torch.nn as nn
from timm.layers import SwiGLUPacked


# ============================================================================
# UNI2-h Model Loading
# ============================================================================

def load_uni2(device: str):
    """
    Load UNI2-h from HuggingFace via timm.
    Architecture: vit_giant_patch14_224, 1536-dim class token output.
    Requires HF login (gated model: MahmoodLab/UNI2-h).
    Full kwargs required — partial kwargs cause resample_abs_pos_embed crash.
    """
    print("Loading UNI2-h model from MahmoodLab/UNI2-h ...")
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True,
    }
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        **timm_kwargs,
    )
    model.eval()
    model.to(device)
    print("UNI2-h loaded successfully.")
    return model


# ============================================================================
# Dataset Class - SASHA-style Hierarchical Patch Loading
# ============================================================================

class Whole_Slide_Bag_UNI2(Dataset):
    """
    Dataset that mimics SASHA's Whole_Slide_Bag_FP hierarchical patch loading.

    For each LR coordinate:
    - Extracts 1 LR patch at patch_level_low_res
    - Extracts 16 HR patches (4x4 grid) at patch_level_high_res
    """

    def __init__(
        self,
        file_path,
        wsi,
        img_transforms=None,
        patch_level_low_res: int = 3,
        patch_level_high_res: int = 1,
        extract_high_res_features: bool = True,
        dataset_name: str = "camelyon16",
        patch_size: int = 256,
        model_input_size: int = 224,
        max_patches: int = -1,
    ):
        self.wsi = wsi
        self.roi_transforms = img_transforms
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.patch_level_low_res = patch_level_low_res
        self.patch_level_high_res = patch_level_high_res
        self.extract_high_res_features = extract_high_res_features
        self.patch_size = patch_size
        self.model_input_size = model_input_size

        with h5py.File(self.file_path, "r") as f:
            self.length = len(f["coords"])
            if max_patches > 0:
                self.length = min(self.length, max_patches)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as hdf5_file:
            coord = hdf5_file["coords"][idx]

        if self.extract_high_res_features:
            if self.dataset_name == "camelyon16":
                scale_factor = 2 ** (self.patch_level_low_res - self.patch_level_high_res)
                step_size = 256
                x_start, y_start = int(coord[0]), int(coord[1])

                high_resolution_imgs = []
                high_resolution_coords = []

                start_time_hr = time.time()
                for x_step in range(scale_factor):
                    for y_step in range(scale_factor):
                        x_curr = x_start + x_step * 2 * step_size
                        y_curr = y_start + y_step * 2 * step_size
                        patch = self.wsi.read_region(
                            (x_curr, y_curr),
                            self.patch_level_high_res,
                            (self.patch_size, self.patch_size),
                        ).convert("RGB")
                        high_resolution_imgs.append(self.roi_transforms(patch))
                        high_resolution_coords.append(np.array([x_curr, y_curr]))
                high_resolution_imgs = torch.stack(high_resolution_imgs)       # (16, 3, 224, 224)
                high_resolution_coords = np.stack(high_resolution_coords)      # (16, 2)
                end_time_hr = time.time()

                start_time_lr = time.time()
                patch = self.wsi.read_region(
                    (x_start, y_start),
                    self.patch_level_low_res,
                    (self.patch_size, self.patch_size),
                ).convert("RGB")
                low_resolution_imgs = self.roi_transforms(patch)                # (3, 224, 224)
                end_time_lr = time.time()

                return {
                    "hr_img": high_resolution_imgs,
                    "hr_coords": high_resolution_coords,
                    "hr_time": torch.tensor(end_time_hr - start_time_hr),
                    "lr_img": low_resolution_imgs,
                    "lr_coords": coord,
                    "lr_time": torch.tensor(end_time_lr - start_time_lr),
                }
            else:
                raise NotImplementedError(f"Dataset {self.dataset_name} not implemented")

        else:
            start_time = time.time()
            patch = self.wsi.read_region(
                (int(coord[0]), int(coord[1])),
                self.patch_level_low_res,
                (self.patch_size, self.patch_size),
            ).convert("RGB")
            end_time = time.time()
            return {
                "img": self.roi_transforms(patch),
                "coord": coord,
                "time": torch.tensor(end_time - start_time),
            }


# ============================================================================
# Feature Extraction with UNI2
# ============================================================================

def extract_uni2_embedding(model, image_tensor, device):
    """
    Extract UNI2-h embedding.

    Args:
        model: UNI2-h model (timm vit_giant_patch14_224, global_pool='token')
        image_tensor: (B, 3, 224, 224)
        device: cuda/cpu

    Returns:
        (B, 1536) float32 embeddings (class token)
    """
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            embedding = model(image_tensor)  # (B, 1536)
    return embedding.float()


def compute_w_loader(loader, uni2_model, verbose=0,
                     extract_high_res_features=True, device=None):
    """
    Process batches and extract raw UNI2-h features in SASHA format.
    No projection is applied; embeddings are saved at their native 1536-dim.

    Returns (if extract_high_res_features):
        hr_features: (N, 16, 1536)
        hr_coords:   (N, 16, 2)
        hr_total_time: float
        lr_features: (N, 1536)
        lr_coords:   (N, 2)
        lr_total_time: float
    Returns (LR-only):
        features: (N, 1536)
        coords:   (N, 2)
        total_time: float
    """
    if verbose > 0:
        print(f"Processing {len(loader)} batches")

    hr_features_list, lr_features_list = [], []
    hr_coords_list, lr_coords_list = [], []
    hr_total_time = lr_total_time = 0.0

    features_list, coords_list = [], []
    total_time = 0.0

    for data in tqdm(loader):
        if extract_high_res_features:
            hr_img    = data["hr_img"]     # (B, 16, 3, 224, 224)
            hr_coords = data["hr_coords"]  # (B, 16, 2)
            lr_img    = data["lr_img"]     # (B, 3, 224, 224)
            lr_coords = data["lr_coords"]  # (B, 2)
            batch_size = hr_img.shape[0]

            # HR pass
            t0 = time.time()
            hr_batch = hr_img.reshape(-1, 3, 224, 224).to(device, non_blocking=True)
            hr_emb  = extract_uni2_embedding(uni2_model, hr_batch, device)  # (B*16, 1536)
            hr_feat = hr_emb.reshape(batch_size, 16, 1536).cpu()            # (B, 16, 1536)
            hr_features_list.append(hr_feat)
            hr_coords_list.append(hr_coords)
            hr_total_time += data["hr_time"].sum().item() + (time.time() - t0)

            # LR pass
            t0 = time.time()
            lr_batch = lr_img.to(device, non_blocking=True)
            lr_emb  = extract_uni2_embedding(uni2_model, lr_batch, device)  # (B, 1536)
            lr_feat = lr_emb.cpu()                                           # (B, 1536)
            lr_features_list.append(lr_feat)
            lr_coords_list.append(lr_coords)
            lr_total_time += data["lr_time"].sum().item() + (time.time() - t0)

        else:
            t0 = time.time()
            batch  = data["img"].to(device, non_blocking=True)
            coords = data["coord"].numpy().astype(np.int32)
            emb    = extract_uni2_embedding(uni2_model, batch, device)  # (B, 1536)
            features_list.append(emb.cpu().numpy())
            coords_list.append(coords)
            total_time += data["time"].sum().item() + (time.time() - t0)

    if extract_high_res_features:
        hr_features = torch.cat(hr_features_list, dim=0).detach().numpy()
        hr_coords   = np.concatenate(hr_coords_list, axis=0)
        lr_features = torch.cat(lr_features_list, dim=0).detach().numpy()
        lr_coords   = np.concatenate(lr_coords_list, axis=0)
        return hr_features, hr_coords, hr_total_time, lr_features, lr_coords, lr_total_time
    else:
        return (
            np.concatenate(features_list, axis=0),
            np.concatenate(coords_list, axis=0),
            total_time,
        )


# ============================================================================
# Main Script
# ============================================================================

def get_arguments():
    parser = argparse.ArgumentParser(description="UNI2-h Feature Extraction for SASHA")

    parser.add_argument("--dataset_name", type=str, default="camelyon16",
                        choices=["camelyon16"])
    parser.add_argument("--data_h5_dir", type=str, required=True,
                        help="Directory containing patches/ subdir with .h5 files from Step 1")
    parser.add_argument("--data_slide_dir", type=str, required=True,
                        help="Directory containing WSI .tif files")
    parser.add_argument("--slide_ext", type=str, default=".tif")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV with columns: case_id, slide_id, label")
    parser.add_argument("--feat_dir", type=str, required=True,
                        help="Output directory (UNI2-only, keep isolated from Virchow outputs)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--extract_high_res_features", type=lambda x: x.lower() != "false",
                        default=True)
    parser.add_argument("--patch_level_low_res", type=int, default=3)
    parser.add_argument("--patch_level_high_res", type=int, default=1)
    parser.add_argument("--target_patch_size", type=int, default=224)

    # Smoke-test / safety flags
    parser.add_argument("--smoke_test", action="store_true",
                        help="Limit to --max_slides slides and --max_patches_per_slide patches")
    parser.add_argument("--max_slides", type=int, default=1,
                        help="Max slides to process (smoke test only)")
    parser.add_argument("--max_patches_per_slide", type=int, default=8,
                        help="Max patches per slide (smoke test only)")

    return parser.parse_args()


class Dataset_All_Bags(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # slide_id in CSV does NOT include .tif
        return self.df.iloc[idx]["slide_id"]


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        print("WARNING: Running on CPU. This will be very slow. Use a GPU node!")

    args = get_arguments()

    print("Initializing dataset")
    bags_dataset = Dataset_All_Bags(args.csv_path)
    df = bags_dataset.df.set_index("slide_id")
    print(f"Total slides in CSV: {len(bags_dataset)}")

    if args.smoke_test:
        print(f"SMOKE TEST MODE: limiting to {args.max_slides} slide(s), "
              f"{args.max_patches_per_slide} patches/slide")

    # Output directories (isolated under feat_dir — never in Virchow paths)
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "lr", "pt_files"), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "lr", "h5_files"), exist_ok=True)

    h5_path_lr = os.path.join(args.feat_dir, "lr", "h5_files", "patch_feats_pretrain_uni2h.h5")
    h5file_lr = h5py.File(h5_path_lr, "w")

    if args.extract_high_res_features:
        os.makedirs(os.path.join(args.feat_dir, "hr", "pt_files"), exist_ok=True)
        os.makedirs(os.path.join(args.feat_dir, "hr", "h5_files"), exist_ok=True)
        h5_path_hr = os.path.join(args.feat_dir, "hr", "h5_files", "patch_feats_pretrain_uni2h.h5")
        h5file_hr = h5py.File(h5_path_hr, "w")

    # Load model
    print("Loading UNI2-h model...")
    uni2_model = load_uni2(device)

    img_transforms = transforms.Compose([
        transforms.Resize(args.target_patch_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    loader_kwargs = {"num_workers": 4, "pin_memory": True} if device.type == "cuda" else {}

    n_slides = len(bags_dataset)
    if args.smoke_test:
        n_slides = min(n_slides, args.max_slides)
    max_patches = args.max_patches_per_slide if args.smoke_test else -1

    processed = 0
    for bag_idx in tqdm(range(n_slides)):
        slide_id = bags_dataset[bag_idx]                    # e.g. "normal_001"
        bag_name = slide_id + ".h5"
        h5_file_path = os.path.join(args.data_h5_dir, "patches", bag_name)

        if not os.path.exists(h5_file_path):
            print(f"Skipping {slide_id}: patch file not found at {h5_file_path}")
            continue

        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        if not os.path.exists(slide_file_path):
            print(f"ERROR: slide not found: {slide_file_path}")
            continue

        print(f"\n[{bag_idx + 1}/{n_slides}] Processing: {slide_id}")
        time_start = time.time()

        wsi = openslide.OpenSlide(slide_file_path)
        dataset = Whole_Slide_Bag_UNI2(
            file_path=h5_file_path,
            wsi=wsi,
            img_transforms=img_transforms,
            extract_high_res_features=args.extract_high_res_features,
            patch_level_low_res=args.patch_level_low_res,
            patch_level_high_res=args.patch_level_high_res,
            dataset_name=args.dataset_name,
            max_patches=max_patches,
        )
        print(f"  Patches to process: {len(dataset)}")

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

        if args.extract_high_res_features:
            hr_features, hr_coords, hr_total_time, lr_features, lr_coords, lr_total_time = \
                compute_w_loader(loader, uni2_model, verbose=1,
                                 device=device, extract_high_res_features=True)

            time_elapsed = time.time() - time_start
            print(f"  Done in {time_elapsed:.2f}s")
            print(f"  HR features shape: {hr_features.shape}  dtype: {hr_features.dtype}")
            print(f"  LR features shape: {lr_features.shape}  dtype: {lr_features.dtype}")

            # Save HR
            grp = h5file_hr.create_group(slide_id)
            grp.create_dataset("feat", data=hr_features.astype(np.float16))
            grp.create_dataset("coords", data=hr_coords.astype(np.int32))
            grp.attrs["label"] = df.loc[slide_id]["label"]

            torch.save(
                torch.from_numpy(hr_features),
                os.path.join(args.feat_dir, "hr", "pt_files", slide_id + ".pt"),
            )

            # Save LR
            grp = h5file_lr.create_group(slide_id)
            grp.create_dataset("feat", data=lr_features.astype(np.float16))
            grp.create_dataset("coords", data=lr_coords.astype(np.int32))
            grp.attrs["label"] = df.loc[slide_id]["label"]

            torch.save(
                torch.from_numpy(lr_features),
                os.path.join(args.feat_dir, "lr", "pt_files", slide_id + ".pt"),
            )

        else:
            features, coords, total_time = compute_w_loader(
                loader, uni2_model, verbose=1,
                device=device, extract_high_res_features=False,
            )

            time_elapsed = time.time() - time_start
            print(f"  Done in {time_elapsed:.2f}s")
            print(f"  Features shape: {features.shape}")

            grp = h5file_lr.create_group(slide_id)
            grp.create_dataset("feat", data=features.astype(np.float16))
            grp.create_dataset("coords", data=coords.astype(np.int32))
            grp.attrs["label"] = df.loc[slide_id]["label"]

            torch.save(
                torch.from_numpy(features),
                os.path.join(args.feat_dir, "lr", "pt_files", slide_id + ".pt"),
            )

        wsi.close()
        processed += 1

    h5file_lr.close()
    if args.extract_high_res_features:
        h5file_hr.close()

    print(f"\nFeature extraction complete! Processed {processed} slide(s).")
    print(f"Outputs saved to: {args.feat_dir}")
    if args.smoke_test:
        print("SMOKE TEST finished — check shapes above before scaling to full subset16.")
