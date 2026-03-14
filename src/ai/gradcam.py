import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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

def get_heatmap(cam, image_tensor):
    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam[0]

def attention_zones(heatmap, raw_image_np, n_peripheral_rings=3):
    h, w = heatmap.shape
    cx, cy = w // 2, h // 2

    # Mask out non-retinal regions using image brightness
    gray = np.mean(raw_image_np, axis=2)
    fov_mask = gray > 0.08

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    fov_pixels = dist[fov_mask]

    if len(fov_pixels) == 0:
        return {}

    max_fov_dist = np.percentile(fov_pixels, 95)
    fundus_radius = max_fov_dist * (45 / 200)  # 45° boundary

    #only count attention within the FOV
    total_attention = heatmap[fov_mask].sum() + 1e-8
    total_fov_area = fov_mask.sum() + 1e-8

    results = {}

    #zone 0: Central 45°
    central_mask = fov_mask & (dist <= fundus_radius)
    central_attn = float(heatmap[central_mask].sum() / total_attention)
    central_area_frac = float(central_mask.sum() / total_fov_area)
    results["fundus_45"] = central_attn
    results["fundus_45_density"] = central_attn / (central_area_frac + 1e-8)

    #zones 1-N: Peripheral rings
    peripheral_width = (max_fov_dist - fundus_radius) / n_peripheral_rings
    for ring in range(n_peripheral_rings):
        inner = fundus_radius + ring * peripheral_width
        outer = fundus_radius + (ring + 1) * peripheral_width
        ring_mask = fov_mask & (dist > inner) & (dist <= outer)
        ring_attn = float(heatmap[ring_mask].sum() / total_attention)
        ring_area_frac = float(ring_mask.sum() / total_fov_area)
        results[f"periph_{ring+1}"] = ring_attn
        results[f"periph_{ring+1}_density"] = ring_attn / (ring_area_frac + 1e-8)

    #outside the retinal fov entirely
    results["artifact"] = float(heatmap[~fov_mask].sum() / (heatmap.sum() + 1e-8))

    return results


def save_overlay(raw_image_np, heatmap, save_path, title="", n_peripheral_rings=3):
    overlay = show_cam_on_image(raw_image_np, heatmap, use_rgb=True)
    h, w = heatmap.shape
    cx, cy = w // 2, h // 2

    #compute zone radii
    gray = np.mean(raw_image_np, axis=2)
    fov_mask = gray > 0.08
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    fov_pixels = dist[fov_mask]
    max_fov_dist = np.percentile(fov_pixels, 95) if len(fov_pixels) > 0 else min(h, w) // 2
    fundus_radius = max_fov_dist * (45 / 200)
    peripheral_width = (max_fov_dist - fundus_radius) / n_peripheral_rings

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_image_np)
    axes[0].set_title("Original")

    axes[1].imshow(heatmap, cmap="jet")

    #draw zone boundaries on heatmap
    zone_colors = ["white", "cyan", "yellow", "red"]
    zone_labels = ["45°", "P1", "P2", "P3"]

    #fundus 45 degree boundary
    axes[1].add_patch(plt.Circle((cx, cy), fundus_radius,
                      fill=False, color=zone_colors[0], linewidth=1.5, linestyle="--"))
    
    #peripheral ring boundaries
    for ring in range(n_peripheral_rings):
        r = fundus_radius + (ring + 1) * peripheral_width
        axes[1].add_patch(plt.Circle((cx, cy), r,
                          fill=False, color=zone_colors[ring + 1], linewidth=1, linestyle=":"))
    axes[1].set_title("GradCAM")

    axes[2].imshow(overlay)
    #draw same boundaries on overlay
    axes[2].add_patch(plt.Circle((cx, cy), fundus_radius,
                      fill=False, color="white", linewidth=1.5, linestyle="--"))
    
    for ring in range(n_peripheral_rings):
        r = fundus_radius + (ring + 1) * peripheral_width
        axes[2].add_patch(plt.Circle((cx, cy), r,
                          fill=False, color="white", linewidth=1, linestyle=":"))
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_gradcam(model, class_names, dataset, resize, val_transform, raw_transform,
                 overlay_output_dir=None, label=""):
    """
    Run GradCAM analysis on the full dataset and return per-class stats.
    Only saves overlays if overlay_output_dir is provided.
    """
    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    class_stats = defaultdict(list)

    for i in tqdm(range(len(dataset)), desc=label):
        pil_img, lbl = dataset[i]
        pil_img = resize(pil_img)

        input_tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)
        raw_np = raw_transform(pil_img).permute(1, 2, 0).numpy()

        heatmap = cam(input_tensor=input_tensor)[0]

        zones = attention_zones(heatmap, raw_np)

        if not zones:
            continue

        with torch.no_grad():
            pred = model(input_tensor).argmax(1).item()

        result = {
            **zones,
            "correct": pred == lbl,
            "true_class": class_names[lbl],
            "pred_class": class_names[pred],
        }
        class_stats[class_names[lbl]].append(result)

        # Save overlays for a few examples per class
        if overlay_output_dir:
            saved_count = sum(1 for r in class_stats[class_names[lbl]] if r.get("saved"))

            if saved_count < 3:
                result["saved"] = True
                save_filename = f"{class_names[lbl]}_{i}.png"
                title = f"True: {class_names[lbl]} | Pred: {class_names[pred]}"
                save_overlay(raw_np, heatmap, Path(overlay_output_dir) / save_filename, title)

    return class_stats


def print_tables(class_stats, class_names, label=""):

    if label:
        print(f"\n{'=' * 64}")
        print(f"  {label}")
        print(f"{'=' * 64}")

    zone_keys = ["fundus_45", "periph_1", "periph_2", "periph_3", "artifact"]
    print(f"\n{'=== Attention Distribution (% of FOV attention) ===':}")
    print(f"{'Class':<12}{'fundus_45':>12}{'periph_1':>10}{'periph_2':>10}{'periph_3':>10}{'artifact':>10}")
    print("-" * 64)
    
    for cls in class_names:
        stats = class_stats[cls]

        if not stats:
            continue
        row = f"{cls:<12}"

        for key in zone_keys:
            val = np.mean([s.get(key, 0) for s in stats])
            row += f"{val:>10.1%}"
        print(row)

    density_keys = ["fundus_45_density", "periph_1_density", "periph_2_density", "periph_3_density"]
    print(f"\n{'=== Attention Density (1.0 = uniform, >1.0 = disproportionate) ===':}")
    print(f"{'Class':<12}{'fundus_45':>12}{'periph_1':>10}{'periph_2':>10}{'periph_3':>10}")
    print("-" * 54)

    for cls in class_names:
        stats = class_stats[cls]

        if not stats:
            continue
        row = f"{cls:<12}"

        for key in density_keys:
            val = np.mean([s.get(key, 0) for s in stats])
            row += f"{val:>10.2f}"
        print(row)

def main():
    #define paths as variables
    model_file_path = "best_model_200deg.pth"
    data_root_path = r"C:\retinal-ai\uwf_images"
    overlay_output_dir = "enhanced_overlays"
    
    #ensure the overlay directory exists
    os.makedirs(overlay_output_dir, exist_ok=True)

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

    #run Gradcam on data
    trained_stats = run_gradcam(
        model, class_names, dataset, resize, val_transform, raw_transform,
        overlay_output_dir=overlay_output_dir, label="Trained model",
    )
    print_tables(trained_stats, class_names, label="TRAINED MODEL")

    #sanity check
    print("\n\nRunning sanity check (randomized classifier)...")
    sanity_model, _ = load_model(model_file_path)
    nn.init.normal_(sanity_model.classifier[1].weight, mean=0.0, std=0.01)
    nn.init.zeros_(sanity_model.classifier[1].bias)

    sanity_stats = run_gradcam(
        sanity_model, class_names, dataset, resize, val_transform, raw_transform,
        overlay_output_dir=None, label="Sanity check", 
    )
    print_tables(sanity_stats, class_names, label="SANITY CHECK (randomized classifier)")

    # Compare random with actual gradcam
    print(f"\n{'=' * 64}")
    print(f"  SANITY CHECK COMPARISON (fundus_45 density)")
    print(f"{'=' * 64}")
    print(f"{'Class':<12}{'Trained':>10}{'Random':>10}{'Diff':>10}")
    print("-" * 42)

    for cls in class_names:
        trained_d = np.mean([s.get("fundus_45_density", 0) for s in trained_stats[cls]])
        sanity_d = np.mean([s.get("fundus_45_density", 0) for s in sanity_stats[cls]])
        diff = trained_d - sanity_d
        print(f"{cls:<12}{trained_d:>10.2f}{sanity_d:>10.2f}{diff:>+10.2f}")

if __name__ == "__main__":
    main()