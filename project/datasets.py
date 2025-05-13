import os
import json
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig


class BSDataset(Dataset):
    """
    Dataset-Klasse für BSData (Binary Classification).
    Ein Bild gilt als 'Schaden', wenn eine JSON-Annotation existiert, sonst als 'kein Schaden'.
    """
    def __init__(self,
                 data_dir: str,
                 label_dir: str,
                 transform: Optional[transforms.Compose] = None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.image_names = sorted(os.listdir(data_dir))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Bild laden
        img_name = self.image_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # JSON-Annotation prüfen
        base = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.label_dir, f"{base}.json")
        has_annotation = os.path.exists(ann_path)

        # Label: 1 = Schaden, 0 = kein Schaden
        label = 1 if has_annotation else 0

        # Transformation anwenden
        if self.transform:
            image = self.transform(image)

        return image, label


class SeverstalDataset(Dataset):
    """
    Dataset-Klasse für Severstal.
    Bisher liefert die Annotation das rohe JSON zurück,
    Label-Extraktion für Multiclass kann später zusätzlich implementiert werden.
    """
    def __init__(self,
                 img_dir: str,
                 ann_dir: str,
                 transform: Optional[transforms.Compose] = None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.image_names = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[dict]]:
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        base = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.ann_dir, f"{base}.json")
        annotation = None
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                annotation = json.load(f)

        if self.transform:
            image = self.transform(image)

        return image, annotation


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Einstiegspunkt zum Erzeugen der DataLoader mittels Hydra-Konfiguration.
    """
    # Transform definieren: Resize auf cfg.data.img_size x img_size
    train_transform = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
    ])

    root = hydra.utils.get_original_cwd()

    # BSData DataLoader (Binary)
    bs_dataset = BSDataset(
        data_dir=os.path.join(root, cfg.data.bsdata.data_dir),
        label_dir=os.path.join(root, cfg.data.bsdata.label_dir),
        transform=train_transform
    )
    bs_loader = torch.utils.data.DataLoader(
        bs_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers
    )

    # Severstal DataLoader (Multiclass)
    sever_dataset = SeverstalDataset(
        img_dir=os.path.join(root, cfg.data.severstal.img_dir),
        ann_dir=os.path.join(root, cfg.data.severstal.ann_dir),
        transform=train_transform
    )
    sever_loader = torch.utils.data.DataLoader(
        sever_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers
    )

    return bs_loader, sever_loader


if __name__ == "__main__":
    main()
