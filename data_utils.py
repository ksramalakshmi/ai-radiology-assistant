import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class XrayDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(sorted(os.listdir(image_dir))):
            class_folder = os.path.join(image_dir, class_name)
            if os.path.isdir(class_folder):
                for img_file in os.listdir(class_folder):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.images.append(os.path.join(class_folder, img_file))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label

def get_loaders(cfg):
    train_aug = A.Compose([
        A.RandomCrop(height=int(cfg['data']['input_size']*0.9), width=int(cfg['data']['input_size']*0.9)),
        A.Resize(height=cfg['data']['input_size'], width=cfg['data']['input_size']),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, value=129, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_aug = A.Compose([
        A.Resize(cfg['data']['input_size'], cfg['data']['input_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = XrayDataset(cfg['data']['train_dir'], transform=train_aug)
    val_dataset = XrayDataset(cfg['data']['val_dir'], transform=val_aug)

    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

    return train_loader, val_loader
