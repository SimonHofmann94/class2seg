import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from project.datasets import BSDataset, SeverstalDataset
from project.utils import set_seed, EarlyStopping

from project.models import ClassifierModel


def get_transforms(img_size: int, augment: bool = True):
    """
    Erzeugt Trainings- und Validierungs-Transforms.
    Data Augmentation wird nur im Training aktiviert.
    """
    train_transforms = []
    if augment:
        # Beispiel-Data-Augmentation
        train_transforms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
    train_transforms += [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_transforms = transforms.Compose(train_transforms)

    return train_transforms, val_transforms


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Seed setzen für Reproduzierbarkeit
    set_seed(cfg.train.seed)

    # TensorBoard-Writer
    tb_log_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.train.tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Transforms inkl. Data Augmentation
    train_tf, val_tf = get_transforms(cfg.data.img_size, cfg.augment.enable)

    # Datasets und Loader
    bs_train = BSDataset(
        data_dir=os.path.join(hydra.utils.get_original_cwd(), cfg.data.bsdata.data_dir),
        label_dir=os.path.join(hydra.utils.get_original_cwd(), cfg.data.bsdata.label_dir),
        transform=train_tf
    )
    bs_val = BSDataset(
        data_dir=os.path.join(hydra.utils.get_original_cwd(), cfg.data.bsdata.data_dir_val),
        label_dir=os.path.join(hydra.utils.get_original_cwd(), cfg.data.bsdata.label_dir_val),
        transform=val_tf
    )
    train_loader = DataLoader(bs_train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(bs_val, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    # Modell: VGG11 oder VGG16
    if cfg.model.backbone == "vgg11":
        model = ClassifierModel(
            backbone_name=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes
        ).to(cfg.train.device)
    else:
        model = ClassifierModel(
            backbone_name=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes
        ).to(cfg.train.device)

    # Klassifikationskopf anpassen
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, cfg.model.num_classes)
    model = model.to(cfg.train.device)

    # Optimizer & LR-Scheduler
    if cfg.train.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.train.lr_patience)

    # Early Stopping
    early_stopper = EarlyStopping(patience=cfg.train.early_stop_patience)

    # Training Loop
    for epoch in range(cfg.train.max_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(cfg.train.device), labels.to(cfg.train.device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(cfg.train.device), labels.to(cfg.train.device)
                outputs = model(imgs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        # LR-Scheduler und EarlyStopping
        scheduler.step(val_loss)
        if early_stopper.step(val_loss):
            print(f"Early stopping bei Epoche {epoch}")
            break

    # Modell speichern
    ckpt_path = os.path.join(hydra.utils.get_original_cwd(), cfg.train.ckpt_dir, f"model_{cfg.model.backbone}.pth")
    torch.save(model.state_dict(), ckpt_path)
    writer.close()


if __name__ == "__main__":
    main()
