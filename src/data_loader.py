import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def main():
    print("-------------EuroSAT dataset------------")
    
    # normalizing/resizing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(), # scaling -> [0.0, 1.0]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    #path to the extracted class directories
    dataset_path = "./data/eurosat/27k/EuroSAT_RGB"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"could not find dataset folder at {dataset_path}")

    print(f"loading image folders from '{dataset_path}'...")
    full_dataset = ImageFolder(root=dataset_path, transform=transform)

    classes = full_dataset.classes
    total_samples = len(full_dataset)
    
    print("\n---------- dataset summary ------------")
    print(f"Total Satellite Image Patches: {total_samples}")
    print(f"Number of Classes: {len(classes)}")
    print(f"Classes: {classes}\n")

    # data split
    train_size = int(0.80 * total_samples)
    val_size = int(0.10 * total_samples)
    test_size = total_samples - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )

    print("------------------ split completed ----------------")
    print(f"train set: {len(train_ds)} samples (80%)")
    print(f"val set:   {len(val_ds)} samples (10%)")
    print(f"test set:  {len(test_ds)} samples (10%)")

    #pyTorch data loaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    
    first_batch, first_labels = next(iter(train_loader))
    print("\n--------------- tensor verification --------------")
    print(f"Batch Image Tensor Shape: {first_batch.shape}  --> (Batch, Channels, Height, Width)")
    print(f"Batch Labels Shape:       {first_labels.shape}")
    print("------------------- completed -----------------\n")

if __name__ == "__main__":
    main()