#!/usr/bin/env python3
"""
🌱 Bean Stalk Damage Identifier
================================
Downloads bean disease images, trains a CNN classifier, and can predict
on new photos. All in one script.

Usage:
    python bean_identifier.py download   # download & organize dataset
    python bean_identifier.py train      # train the model
    python bean_identifier.py predict <image.jpg>  # classify a photo
    python bean_identifier.py all        # do all three in sequence
"""

import argparse
import sys
import os
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG — tweak these if you like
# ──────────────────────────────────────────────
DATA_DIR = Path("data")
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
MODEL_PATH = "model.pth"

# Classes we'll train on (folder names under data/train/)
CLASS_NAMES = ("healthy", "rust", "bacterial blight")

# ──────────────────────────────────────────────
# 1. DOWNLOAD — pulls bean images from Hugging Face
# ──────────────────────────────────────────────
def download():
    print("=" * 55)
    print("📥 Step 1: Downloading bean disease dataset from PlantVillage")
    print("=" * 55)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' library...")
        os.system("pip install datasets -q")
        from datasets import load_dataset

    ds = load_dataset("plantvillage/plantvillage", split="train", trust_remote_code=True)

    # Filter for bean classes only
    hf_labels = ("Bean___healthy", "Bean___rust", "Bean___bacterial_blight")
    bean_ds = ds.filter(lambda x: x["label_name"] in hf_labels)

    print(f"  Found {len(bean_ds)} bean images total")

    # Create folder structure
    for split in ("train", "val"):
        for cls in CLASS_NAMES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # Split 80/20 train/val and save images
    saved = {"train": 0, "val": 0}
    for i, sample in enumerate(bean_ds):
        img = sample["image"]
        label = sample["label_name"]
        folder = label.replace("Bean___", "").replace("_", " ")

        split = "train" if i % 5 != 0 else "val"
        fname = f"{i:05d}.jpg"
        img.save(DATA_DIR / split / folder / fname)
        saved[split] += 1

    print(f"  ✅ Saved: {saved['train']} train + {saved['val']} val images")
    print(f"  📁 Location: ./data/train/ and ./data/val/")
    print(f"  🏷️ Classes: {CLASS_NAMES}\n")

# ──────────────────────────────────────────────
# 2. TRAIN — builds and trains the model
# ──────────────────────────────────────────────
def train():
    print("=" * 55)
    print("🧠 Step 2: Training the model")
    print("=" * 55)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Count actual classes from folders
    train_path = DATA_DIR / "train"
    actual_classes = sorted(d.name for d in train_path.iterdir() if d.is_dir())
    num_classes = len(actual_classes)
    print(f"  Found {num_classes} classes: {actual_classes}")

    # Transforms
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfm)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tfm)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model — EfficientNet-B0 with pretrained weights
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"    ✅ Saved best model ({val_acc:.1f}%)")

    print(f"\n  🏁 Training complete! Best val accuracy: {best_acc:.1f}%")
    print(f"  💾 Model saved to: {MODEL_PATH}\n")

# ──────────────────────────────────────────────
# 3. PREDICT — classify a single image
# ──────────────────────────────────────────────
def predict(image_path):
    print("=" * 55)
    print(f"🔍 Step 3: Predicting on {image_path}")
    print("=" * 55)

    import torch
    from torchvision import transforms, models
    from torch import nn
    from PIL import Image

    if not Path(image_path).exists():
        print(f"  ❌ File not found: {image_path}")
        sys.exit(1)

    if not Path(MODEL_PATH).exists():
        print(f"  ❌ No trained model found at {MODEL_PATH}")
        print(f"  Run 'python bean_identifier.py train' first")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild model architecture
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probs).item()

    confidence = probs[predicted_idx].item() * 100
    predicted_class = CLASS_NAMES[predicted_idx]

    print(f"\n  📸 Image: {image_path}")
    print(f"  🏷️ Prediction: {predicted_class}")
    print(f"  📊 Confidence: {confidence:.1f}%")
    print(f"\n  📋 Breakdown:")
    for i, cls in enumerate(CLASS_NAMES):
        bar = "█" * int(probs[i].item() * 30)
        print(f"    {cls:20s} {probs[i].item()*100:5.1f}% {bar}")
    print()

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="🌱 Bean Stalk Damage Identifier")
    parser.add_argument("mode", choices=["download", "train", "predict", "all"],
                         help="What to do")
    parser.add_argument("image", nargs="?", help="Path to image (for predict mode)")
    args = parser.parse_args()

    if args.mode == "download":
        download()

    elif args.mode == "train":
        # Check data exists
        if not (DATA_DIR / "train").exists() or not any((DATA_DIR / "train").iterdir()):
            print("⚠️ No training data found. Run 'download' first or place images in data/train/")
            sys.exit(1)
        train()

    elif args.mode == "predict":
        if not args.image:
            print("❌ Please provide an image path: python bean_identifier.py predict photo.jpg")
            sys.exit(1)
        predict(args.image)

    elif args.mode == "all":
        print("\n🌱 BEAN STALK DAMAGE IDENTIFIER — Full Pipeline\n")
        download()
        train()
        print("✅ All done! Now you can classify images:")
        print(f"  python bean_identifier.py predict path/to/bean_photo.jpg\n")

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────
# USAGE (run these in a terminal, not as part of the script):
#
# 1. Install dependencies (one time)
#    pip install torch torchvision pillow datasets
#
# 2. Do everything — download, train, and you're ready
#    python bean_identifier.py all
#
# 3. Classify a photo of a bean stalk
#    python bean_identifier.py predict your_bean_photo.jpg
# ──────────────────────────────────────────────
