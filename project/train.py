import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import hydra
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as transforms
from omegaconf import DictConfig
import torch.nn.functional as F

from new.project_old import BSDataset
from new.project_old import ClassifierModel
from new.project_old import set_seed, EarlyStopping


def get_transforms(cfg: DictConfig):
    img_size = cfg.data.img_size
    aug      = cfg.augment

    # --- 1) PIL-Augmentations ---
    pil_transforms = []
    if aug.use_resizedcrop:
        pil_transforms.append(
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            )
        )
    if aug.use_hflip:
        pil_transforms.append(transforms.RandomHorizontalFlip())
    if aug.use_vflip:
        pil_transforms.append(transforms.RandomVerticalFlip())
    if aug.use_rotation:
        pil_transforms.append(transforms.RandomRotation(aug.rotation_degree))
    if aug.use_colorjitter:
        pil_transforms.append(
            transforms.ColorJitter(
                brightness=aug.cj_brightness,
                contrast=aug.cj_contrast,
                saturation=aug.cj_saturation,
                hue=aug.cj_hue
            )
        )
    if aug.use_perspective:
        pil_transforms.append(
            transforms.RandomPerspective(
                distortion_scale=aug.perspective_distortion_scale,
                p=aug.perspective_p
            )
        )

    # immer noch PIL: Bild auf endgültige Größe bringen
    pil_transforms.append(transforms.Resize((img_size, img_size)))
    # jetzt Konvertierung zu Tensor
    pil_transforms.append(transforms.ToTensor())

    # --- 2) Tensor-Augmentations & Normalisierung ---
    tensor_transforms = []
    if aug.use_blur:
        tensor_transforms.append(
            transforms.GaussianBlur(
                kernel_size=tuple(aug.blur_kernel),
                sigma=tuple(aug.blur_sigma)
            )
        )
    if aug.use_erasing:
        tensor_transforms.append(
            transforms.RandomErasing(
                p=aug.erasing_p,
                scale=tuple(aug.erasing_scale),
                ratio=tuple(aug.erasing_ratio)
            )
        )
    # ganz am Schluss normalisieren
    tensor_transforms.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        )
    )

    train_tf = transforms.Compose(pil_transforms + tensor_transforms)

    # Für Validation nur Resize→ToTensor→Normalize
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_tf, val_tf

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    inputs: rohe Logits (Batch x Classes)
    targets: LongTensor mit Labels (Batch,)
    """
    ce = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)  # pt = softmax-Wahrscheinlichkeit der korrekten Klasse
    focal = alpha * (1 - pt)**gamma * ce
    return focal.mean() if reduction=='mean' else focal.sum()


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Seeds & Device
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    # TensorBoard
    tb_log_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.train.tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Transforms
    train_tf, val_tf = get_transforms(cfg)

    # Pfade
    root = hydra.utils.get_original_cwd()

    # --- BSData: laden und splitten ---
    # 1) Volles Dataset (BSData) laden
    full_bs = BSDataset(
        data_dir=os.path.join(root, cfg.data.bsdata.data_dir),
        label_dir=os.path.join(root, cfg.data.bsdata.label_dir),
        transform=train_tf
    )
    n_total = len(full_bs)

    # 2) Test-Set abtrennen
    n_test = int(n_total * cfg.data.bsdata.test_split)
    n_rest = n_total - n_test
    bs_rest, bs_test = random_split(
        full_bs,
        [n_rest, n_test],
        generator=torch.Generator().manual_seed(cfg.train.seed)
    )

    # 3) Rest in Train & Val splitten
    n_train = int(n_rest * cfg.data.bsdata.train_val_split)
    n_val = n_rest - n_train
    bs_train, bs_val = random_split(
        bs_rest,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.train.seed)
    )

    # 4) Oversampling nur auf das Trainings-Set anwenden
    train_labels = [label for _, label in bs_train]
    class_counts = [train_labels.count(c) for c in range(cfg.model.num_classes)]
    class_weights = [1.0 / count if count > 0 else 0.0 for count in class_counts]
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 5) DataLoader anlegen
    train_loader = DataLoader(
        bs_train,
        batch_size=cfg.train.batch_size,
        sampler=sampler,  # hier statt shuffle=True
        num_workers=cfg.train.num_workers
    )
    val_loader = DataLoader(
        bs_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers
    )
    test_loader = DataLoader(
        bs_test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers
    )

    # --- Class-Weights berechnen und Criterion setzen ---
    # Label-Liste extrahieren (0/1)
    labels = [label for _, label in bs_train]
    counts = [labels.count(i) for i in range(cfg.model.num_classes)]
    total = sum(counts)
    # weights invers proportional zur Häufigkeit
    weights = [total / c if c > 0 else 0.0 for c in counts]
    weight_tensor = torch.tensor(weights, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # --- Modell initialisieren ---
    model = ClassifierModel(
        backbone_name=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr) if cfg.train.optimizer == "adam" \
        else optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.train.lr_patience)

    # EarlyStopping
    early_stopper = EarlyStopping(patience=cfg.train.early_stop_patience)

    # --- Trainings-Loop ---
    for epoch in range(cfg.train.max_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = focal_loss(outputs, labels, alpha=0.25, gamma=2.0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar("Loss/Train", epoch_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        # Scheduler & Early Stopping
        scheduler.step(val_loss)
        if early_stopper.step(val_loss):
            print(f"Early stopping bei Epoche {epoch}")
            break

    # Checkpoint
    ckpt_path = os.path.join(root, cfg.train.ckpt_dir, f"model_{cfg.model.backbone}.pth")
    torch.save(model.state_dict(), ckpt_path)
    writer.close()


if __name__ == "__main__":
    main()
