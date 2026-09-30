import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RUNS = Path(__file__).resolve().parent
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CIFAR_CLASSES =[
    "airplane", "automobile", "bird", "cat", "deer", "dog",
    "frog", "horse", "ship", "truck"
]


def load(tag):
    with open(RUNS / f"{tag}.pkl", "rb") as f:
        return pickle.load(f)


def report(tag):
    d = load(tag)
    dataset = d.get("dataset", "mnist")
    fashion = dataset in ("fashion", "fashion-mnist", "fashion_mnist")
    cifar = dataset in ("cifar", "cifar10", "cifar-10")
    family = "CIFAR-10" if cifar else "FashionMNIST" if fashion else "MNIST"
    classes = CIFAR_CLASSES if cifar else FASHION_CLASSES
    class0 = classes[d["d0"]] if fashion or cifar else str(d["d0"])
    class1 = classes[d["d1"]] if fashion or cifar else str(d["d1"])
    print(f"\n=== {tag} ({family}: {class0} -> {class1}) ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    regimes = [("a", "independent init"), ("b", "shared init"), ("c", "warm-start")]

    for ax, (key, title) in zip(axes, regimes):
        rows = d[f"dist_{key}"]
        dt = np.array([r["dt"] for r in rows])
        raw = np.array([r["raw"] for r in rows])
        aligned = np.array([r["aligned"] for r in rows])
        func = np.array([r["functional"] for r in rows])

        rho_raw, _ = spearmanr(dt, raw)
        rho_aligned, _ = spearmanr(dt, aligned)
        rho_func, _ = spearmanr(dt, func)
        rho_af, _ = spearmanr(aligned, func)

        ax.scatter(dt, raw, s=14, alpha=0.5, label=f"raw (ρ={rho_raw:.2f})", color="tab:red")
        ax.scatter(dt, aligned, s=14, alpha=0.5, label=f"aligned (ρ={rho_aligned:.2f})", color="tab:blue")
        ax.set_xlabel(f"|Δt|  ({family} image distance)")
        ax.set_ylabel("weight distance")
        ax.set_title(f"regime ({key}): {title}")
        ax.legend(fontsize=8)

        print(f"  regime {key:1s} | Spearman(dt, raw weight)     = {rho_raw:+.3f}")
        print(f"          | Spearman(dt, aligned weight) = {rho_aligned:+.3f}")
        print(f"          | Spearman(dt, functional)     = {rho_func:+.3f}")
        print(f"          | Spearman(aligned, functional)= {rho_af:+.3f}  (does aligned weight dist track actual image diff?)")

    plt.tight_layout()
    out_path = RUNS / f"{tag}_distance_vs_dt.png"
    plt.savefig(out_path, dpi=130)
    print(f"  saved {out_path}")

    if "image0" in d and "image1" in d or "image2" in d:
        fig0, axes0 = plt.subplots(1, 3, figsize=(8, 3))
        images = [d["image0"], (d["image0"] + d["image1"]) / 2, d["image1"]]
        labels = [f"t=0: {class0}", "t=0.5: morph", f"t=1: {class1}"]
        for ax, img, label in zip(axes0, images, labels):
            ax.imshow(img.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(label)
            ax.axis("off")
        fig0.suptitle(f"{family} input pair and midpoint")
        plt.tight_layout()
        morph_path = RUNS / f"{tag}_morph.png"
        plt.savefig(morph_path, dpi=130)
        print(f"  saved {morph_path}")

    # basin-jump detector on regime (c) consecutive distances
    consec = d["consec_c"]
    t_c = np.array([r["t"] for r in consec])
    aligned_c = np.array([r["aligned"] for r in consec])
    med = np.median(aligned_c)
    mad = np.median(np.abs(aligned_c - med)) + 1e-8
    z = (aligned_c - med) / mad
    jump_idx = np.where(z > 3)[0]
    print(f"  basin-jump candidates (aligned dist MAD-z > 3): t ≈ {np.round(t_c[jump_idx], 3).tolist()}")

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(t_c, aligned_c, marker="o", ms=3)
    for i in jump_idx:
        ax2.axvline(t_c[i], color="red", alpha=0.4, linestyle="--")
    ax2.set_xlabel("t (along warm-started sweep)")
    ax2.set_ylabel("aligned weight dist to previous t")
    ax2.set_title(f"{tag}: basin-jump trace (regime c)")
    plt.tight_layout()
    out_path2 = RUNS / f"{tag}_basin_jumps.png"
    plt.savefig(out_path2, dpi=130)
    print(f"  saved {out_path2}")


if __name__ == "__main__":
    tags = sys.argv[1:] if len(sys.argv) > 1 else [
        "easy_1_1_mnist", "easy_1_1_fashionmnist",
        "easy_1_1_cifar10",
        "hard_3_8_mnist", "hard_3_8_fashionmnist",
        "hard_3_8_cifar10",
    ]
    for t in tags:
        report(t)
