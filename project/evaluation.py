import os
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

from project.datasets import BSDataset
from project.models import ClassifierModel
from project.utils import set_seed

from sklearn.metrics import confusion_matrix


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Evaluation-Skript für das Klassifikationsmodell (BSData binary) mittels random-split.
    Berechnet Precision, Recall, F1 und ROC AUC auf dem Validierungs-Split von BSData.
    """
    # Reproduzierbarkeit
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    # Pfade
    root = hydra.utils.get_original_cwd()
    data_dir  = os.path.join(root, cfg.data.bsdata.data_dir)
    label_dir = os.path.join(root, cfg.data.bsdata.label_dir)
    ckpt_path = os.path.join(root, cfg.train.ckpt_dir, f"model_{cfg.model.backbone}.pth")

    # Transform für Validation
    val_tf = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 1) Volles Dataset laden (mit val_tf)
    full_bs = BSDataset(
        data_dir=data_dir,
        label_dir=label_dir,
        transform=val_tf
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

    # (keinen weiteren Split auf bs_rest nötig, wir haben ja schon train/val in train.py)
    # 3) Nur den Test-Loader anlegen
    test_loader = DataLoader(
        bs_test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers
    )

    # Modell laden
    model = ClassifierModel(
        backbone_name=cfg.model.backbone,
        pretrained=False,
        num_classes=cfg.model.num_classes
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Nur binary unterstützt
    if cfg.model.num_classes != 2:
        print("Multiclass-Evaluation noch nicht implementiert. (Nur binary unterstützt.)")
        return

    # Inferenz
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs[:, 1].cpu().numpy().tolist())


    # Wahrscheinlichkeiten ausgeben
    probs_arr = np.array(y_prob)
    print("Wahrscheinlichkeiten Schaden-Klasse:")
    print("  min:", probs_arr.min(),
          " max:", probs_arr.max(),
          " mean:", probs_arr.mean())

    # Metriken berechnen
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    roc_auc = roc_auc_score(y_true, y_prob)

    # Ergebnisse ausgeben
    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()






