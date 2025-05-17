import os
import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix

from project.datasets import BSDataset
from project.models import ClassifierModel
from project.utils import set_seed


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Evaluation-Skript für das Klassifikationsmodell (BSData binary) auf dem echten Test-Set.
    """
    # Reproduzierbarkeit
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    # Pfade
    root = hydra.utils.get_original_cwd()
    data_dir  = os.path.join(root, cfg.data.bsdata.data_dir)
    label_dir = os.path.join(root, cfg.data.bsdata.label_dir)
    ckpt_path = os.path.join(root, cfg.train.ckpt_dir, f"model_{cfg.model.backbone}.pth")

    # Transform für Test-Set
    test_tf = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset laden und echten Test-Split abtrennen
    full_bs = BSDataset(
        data_dir=data_dir,
        label_dir=label_dir,
        transform=test_tf
    )
    n_total = len(full_bs)
    n_test  = int(n_total * cfg.data.bsdata.test_split)
    n_rest  = n_total - n_test
    _, bs_test = random_split(
        full_bs,
        [n_rest, n_test],
        generator=torch.Generator().manual_seed(cfg.train.seed)
    )
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

    # Nur binary supported
    if cfg.model.num_classes != 2:
        print("Multiclass-Evaluation noch nicht implementiert. (Nur binary unterstützt.)")
        return

    # Inferenz auf Test-Set
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs[:, 1].cpu().numpy().tolist())

    # Metriken berechnen
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    roc_auc = roc_auc_score(y_true, y_prob)
    cm      = confusion_matrix(y_true, y_pred)

    # Ergebnisse ausgeben
    print("\n=== Test Set Evaluation Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
