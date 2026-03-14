import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, checkpoint["class_names"]

def compute_fov_radius(raw_image_np):
    h, w = raw_image_np.shape[:2]
    cx, cy = w // 2, h // 2

    gray = np.mean(raw_image_np, axis=2)
    fov_mask = gray > 0.08

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    fov_pixels = dist[fov_mask]

    if len(fov_pixels) == 0:
        return min(h, w) // 2

    return np.percentile(fov_pixels, 95)

def create_zone_masks(h, w, max_fov_dist, n_peripheral_rings=3):
    cx, cy = w // 2, h // 2
    fundus_radius = max_fov_dist * (45 / 200)
    peripheral_width = (max_fov_dist - fundus_radius) / n_peripheral_rings

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    masks = {}
    masks["fundus_45"] = dist <= fundus_radius

    for ring in range(n_peripheral_rings):
        inner = fundus_radius + ring * peripheral_width
        outer = fundus_radius + (ring + 1) * peripheral_width
        masks[f"periph_{ring+1}"] = (dist > inner) & (dist <= outer)

    return masks

def apply_occlusion(image_tensor, mask, fill_value=0.0):

    occluded = image_tensor.clone()
    mask_tensor = torch.from_numpy(mask).to(image_tensor.device).unsqueeze(0).unsqueeze(0)
    mask_tensor = mask_tensor.expand_as(occluded).float()

    #replace with normalized "black" 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image_tensor.device)
    black_normalized = (0.0 - mean) / std  # What "black" looks like after normalization

    occluded = occluded * mask_tensor + black_normalized * (1 - mask_tensor)

    return occluded


@torch.no_grad()
def evaluate_with_occlusion(model, images, labels, raw_images, occlusion_type, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    #confidence on true class
    all_correct_conf = []

    for img_tensor, label, raw_np in zip(images, labels, raw_images):
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        h, w = img_tensor.shape[2], img_tensor.shape[3]

        max_fov_dist = compute_fov_radius(raw_np)
        zone_masks = create_zone_masks(h, w, max_fov_dist)

        #determine which mask to apply
        if occlusion_type == "none":
            occluded = img_tensor

        elif occlusion_type.startswith("mask_periph_"):
            #mask one specific peripheral ring, keep everything else
            ring = occlusion_type  
            ring_key = ring.replace("mask_", "")  
            keep_mask = ~zone_masks[ring_key] 
            occluded = apply_occlusion(img_tensor, keep_mask)

        else:
            occluded = img_tensor

        output = model(occluded)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()

        all_preds.append(pred)
        all_labels.append(label)
        all_correct_conf.append(probs[0, label].item())

    return np.array(all_labels), np.array(all_preds), np.array(all_correct_conf)

#visualize Occlusion on test samples
def save_occlusion_examples(images, raw_images, labels, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_classes = set()

    for img_tensor, raw_np, label in zip(images, raw_images, labels):
        cls = class_names[label]

        if cls in saved_classes:
            continue
        saved_classes.add(cls)

        h, w = 224, 224
        max_fov_dist = compute_fov_radius(raw_np)
        zone_masks = create_zone_masks(h, w, max_fov_dist)

        img_t = img_tensor.unsqueeze(0).to(DEVICE)

        occlusions = {
            "Full Image": "none",
            "Mask R1": "mask_periph_1",
            "Mask R2": "mask_periph_2",
            "Mask R3": "mask_periph_3",
        }

        fig, axes = plt.subplots(1, len(occlusions), figsize=(3 * len(occlusions), 3))
        fig.suptitle(f"Occlusion Examples — {cls}", fontweight="bold")

        for ax, (title, occ_type) in zip(axes, occlusions.items()):

            if occ_type == "none":
                vis = raw_np

            elif occ_type.startswith("mask_periph_"):
                ring_key = occ_type.replace("mask_", "")
                vis = raw_np * (~zone_masks[ring_key])[:, :, None]

            else:
                vis = raw_np

            ax.imshow(np.clip(vis, 0, 1))
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"occlusion_example_{cls}.png", dpi=150, bbox_inches="tight")
        plt.close()

        if len(saved_classes) == len(class_names):
            break

def main():
    model_file_path = "best_model_200deg.pth"
    data_root_path = r"C:\retinal-ai\uwf_images"
    output_dir = "occlusion_results"

    os.makedirs(output_dir, exist_ok=True)

    model, class_names = load_model(model_file_path)

    dataset = datasets.ImageFolder(data_root_path)
    resize = transforms.Resize((224, 224))
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    raw_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    #preload all images
    print("Preloading images...")
    all_images = []     
    all_raw = []         
    all_labels = []

    for i in tqdm(range(len(dataset)), desc="Loading"):
        pil_img, label = dataset[i]
        pil_img = resize(pil_img)

        all_images.append(val_transform(pil_img))
        all_raw.append(raw_transform(pil_img).permute(1, 2, 0).numpy())
        all_labels.append(label)

    #save example visualizations
    print("Saving occlusion examples...")
    save_occlusion_examples(all_images, all_raw, all_labels, class_names, output_dir)

    #define occlusion conditions
    conditions = [
        ("none", "Baseline"),
        ("mask_periph_1", "Mask Ring 1"),
        ("mask_periph_2", "Mask Ring 2"),
        ("mask_periph_3", "Mask Ring 3"),
    ]

    #run each condition
    all_results = {}

    for occ_type, description in conditions:
        print(f"\nEvaluating: {description}...")
        labels, preds, confs = evaluate_with_occlusion(
            model, all_images, all_labels, all_raw, occ_type, class_names,
        )

        all_results[occ_type] = {
            "labels": labels,
            "preds": preds,
            "confs": confs,
            "description": description,
        }

    #print accuracies
    print(f"\n{'=' * 90}")
    print(f"  Accuracy Drop from Baseline (negative = occlusion hurts)")
    print(f"{'=' * 90}")

    baseline = all_results["none"]

    header = f"{'Class':<12}"

    for occ_type, desc in conditions[1:]:
        short = desc.split("(")[0].strip()[:12]
        header += f"{short:>13}"

    print(header)
    print("-" * (12 + 13 * (len(conditions) - 1)))

    for cls_idx, cls in enumerate(class_names):
        row = f"{cls:<12}"
        cls_mask = baseline["labels"] == cls_idx

        if cls_mask.sum() == 0:
            continue
        base_acc = accuracy_score(baseline["labels"][cls_mask], baseline["preds"][cls_mask])

        for occ_type, _ in conditions[1:]:
            r = all_results[occ_type]
            occ_acc = accuracy_score(r["labels"][cls_mask], r["preds"][cls_mask])
            drop = occ_acc - base_acc
            row += f"{drop:>+13.1%}"

        print(row)

    row = f"{'OVERALL':<12}"
    base_acc = accuracy_score(baseline["labels"], baseline["preds"])

    for occ_type, _ in conditions[1:]:
        r = all_results[occ_type]
        occ_acc = accuracy_score(r["labels"], r["preds"])
        drop = occ_acc - base_acc
        row += f"{drop:>+13.1%}"

    print(row)

    print(f"\n{'=' * 90}")
    print(f"  Mean Confidence on True Class by Occlusion")
    print(f"{'=' * 90}")

    header = f"{'Class':<12}"

    for occ_type, desc in conditions:
        short = desc.split("(")[0].strip()[:12]
        header += f"{short:>13}"

    print(header)
    print("-" * (12 + 13 * len(conditions)))

    for cls_idx, cls in enumerate(class_names):
        row = f"{cls:<12}"

        for occ_type, _ in conditions:
            r = all_results[occ_type]
            cls_mask = r["labels"] == cls_idx

            if cls_mask.sum() == 0:
                row += f"{'N/A':>13}"
                continue

            conf = np.mean(r["confs"][cls_mask])
            row += f"{conf:>13.3f}"
        print(row)

if __name__ == "__main__":
    main()