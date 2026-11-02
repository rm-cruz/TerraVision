import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class EuroSATBaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EuroSATBaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_baseline(epochs=3):
    print("=== Executing 3-Epoch Baseline Training Run ===")
    
    # Explicitly enforce MPS for Mac hardware acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚡ Hardware Acceleration: Enabled (Apple Silicon GPU / MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("⚡ Hardware Acceleration: Enabled (NVIDIA CUDA)")
    else:
        device = torch.device("cpu")
        print("Hardware Acceleration: CPU Mode")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = "./data/eurosat/27k/EuroSAT_RGB"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print("Loading dataset structure...")
    ds = ImageFolder(root=dataset_path, transform=transform)
    train_sz, val_sz = int(0.80 * len(ds)), int(0.10 * len(ds))
    test_sz = len(ds) - train_sz - val_sz
    train_ds, val_ds, _ = random_split(
        ds, 
        [train_sz, val_sz, test_sz], 
        generator=torch.Generator().manual_seed(42)
    )

    # Increased batch_size to 128 for faster memory throughput
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = EuroSATBaselineCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training on {len(train_ds)} samples across {len(train_loader)} batches...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            if batch_idx % 25 == 0 or batch_idx == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}]")

        train_acc = 100.0 * correct / total
        epoch_loss = running_loss / total

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        print(f"--> Epoch [{epoch}/{epochs}] Finished | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%\n")

    torch.save(model.state_dict(), "baseline_cnn.pth")
    print("Model weights saved to 'baseline_cnn.pth' successfully!")

if __name__ == "__main__":
    train_baseline(epochs=3)

