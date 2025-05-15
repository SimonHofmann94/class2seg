import os
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from captum.attr import LayerLRP, IntegratedGradients, Saliency, DeepLift, DeepLiftShap, LayerGradCam
import matplotlib.pyplot as plt

from new.project_old import BSDataset
from new.project_old import ClassifierModel
from new.project_old import set_seed


def get_test_loader(cfg: DictConfig):
    root = hydra.utils.get_original_cwd()
    data_dir = os.path.join(root, cfg.data.bsdata.data_dir)
    label_dir = os.path.join(root, cfg.data.bsdata.label_dir)
    tf = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    full = BSDataset(data_dir, label_dir, transform=tf)
    n = len(full)
    n_test = int(n * cfg.data.bsdata.test_split)
    n_rest = n - n_test
    _, bs_test = random_split(full, [n_rest, n_test], generator=torch.Generator().manual_seed(cfg.train.seed))
    loader = DataLoader(bs_test, batch_size=1, shuffle=False)
    return loader


def visualize_attr(attr_map, input_tensor, save_path):
    """Visuelle Darstellung einer einzelnen Attributionskarte neben dem Input-Bild,
    mit Unnormalisierung und Upsampling auf Input-Auflösung."""
    # Unnormalize für Input-Bild
    mean = torch.tensor([0.485, 0.456, 0.406])[ :, None, None ]
    std  = torch.tensor([0.229, 0.224, 0.225])[ :, None, None ]
    inp_tensor = input_tensor.squeeze().cpu()
    inp_unnorm = inp_tensor * std + mean
    inp = inp_unnorm.permute(1, 2, 0).numpy()

    # Attribution-Map verarbeiten und auf Input-Größe upsamplen
    if isinstance(attr_map, torch.Tensor):
        if attr_map.dim() == 4:
            attr = torch.sum(attr_map, dim=1, keepdim=True)
            attr = F.interpolate(
                attr,
                size=(inp_tensor.shape[1], inp_tensor.shape[2]),
                mode='bilinear',
                align_corners=False
            )
            attr = attr.squeeze().cpu().detach().numpy()
        else:
            attr = attr_map.squeeze().cpu().detach().numpy()
    else:
        attr = attr_map

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(inp)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    im = ax[1].imshow(attr, cmap='hot', interpolation='nearest')
    ax[1].set_title('Attribution')
    ax[1].axis('off')
    fig.colorbar(im, ax=ax[1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Erzeuge XAI-Attributionskarten für das Test-Set und speichere sie."""
    # → 1) Reproduzierbarkeit & Device
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    # ==== 0) XAI-Config auslesen ====
    # Multi-Class Liste, Baseline-Typen und positives-only Flag
    target_classes = cfg.xai.get('target_classes', [cfg.xai.target_class])
    baseline_types  = cfg.xai.get('baseline_types', [])
    positive_only   = cfg.xai.get('positive_only', True)

    # ==== 1) Modell aus Checkpoint laden ====
    model = ClassifierModel(
        backbone_name=cfg.model.backbone,
        pretrained=False,
        num_classes=cfg.model.num_classes
    ).to(device)
    # Pfad zum gespeicherten Modell
    ckpt = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.train.ckpt_dir,
        f"model_{cfg.model.backbone}.pth"
    )
    # Zustands-Dict laden
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Für LRP/GradCam: letzte Identity-Layers entfernen
    try:
        from torch import nn
        children = list(model.features.children())
        # Identität am Ende von features entfernen
        if isinstance(children[-1], nn.Identity):
            model.features = nn.Sequential(*children[:-1])
        # **Avgpool** nur dann anpassen, wenn es wirklich eine Identity ist
        if hasattr(model, 'avgpool') and isinstance(model.avgpool, nn.Identity):
            # bei ResNet/VGG evtl. nötig – sonst nichts tun
            pass
    except Exception:
        pass

    # ==== 2) Test-Loader vorbereiten ====
    loader = get_test_loader(cfg)  # gibt nun (img, label, base) zurück

    # ==== 3) Dataset-Mean Baseline berechnen ====
    mean_baseline = None
    if 'dataset_mean' in baseline_types:
        # Summiere alle Bilder und teile durch Anzahl
        cumulative = torch.zeros_like(next(iter(loader))[0].to(device))
        count = 0
        for img, _, _ in loader:
            cumulative += img.to(device)
            count += 1
        mean_baseline = (cumulative / count).to(device)

    # ==== 4) Output-Verzeichnis ====
    out_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.output.attribution_dir,
        cfg.model.backbone,
        cfg.xai.method.lower()
    )
    os.makedirs(out_dir, exist_ok=True)

    # ==== 5) Attributions generieren ====
    for idx, (img, label, base) in enumerate(loader):
        if idx >= cfg.xai.max_samples:
            break
        img = img.to(device)

        # Für jede Ziel-Klasse in der Liste
        for cls in target_classes:
            method = cfg.xai.method.lower()

            # → Explainer auswählen
            if method == 'lrp':
                convs = [m for m in model.features.modules() if isinstance(m, torch.nn.Conv2d)]
                explainer = LayerLRP(model, convs[-1])
                attr_kwargs = {'target': cls}
            elif method == 'integrated_gradients':
                explainer = IntegratedGradients(model)
                attr_kwargs = {'target': cls}
            elif method == 'saliency':
                # Saliency benötigt ebenfalls einen target-Index
                explainer = Saliency(model)
                attr_kwargs = {'target': cls}
            elif method == 'deeplift':
                explainer = DeepLift(model)
                attr_kwargs = {}
            elif method == 'deeplift_shap':
                explainer = DeepLiftShap(model)
                attr_kwargs = {}
            elif method in ['gradcam', 'gradcam++']:
                # GradCam braucht auch target für multi-class
                convs = [m for m in model.features.modules() if isinstance(m, torch.nn.Conv2d)]
                explainer = LayerGradCam(model, convs[-1])
                attr_kwargs = {'target': cls}
            else:
                raise ValueError(f"Unbekannte XAI-Methode: {cfg.xai.method}")

            # → DeepLift(Shap): für jede Baseline-Typ eine Map
            if method in ['deeplift', 'deeplift_shap'] and baseline_types:
                for btype in baseline_types:
                    if btype == 'zero':
                        baselines = torch.zeros_like(img)
                    elif btype == 'gaussian':
                        baselines = torch.randn_like(img)
                    elif btype == 'dataset_mean':
                        baselines = mean_baseline
                    else:
                        continue

                    at_map = explainer.attribute(img, baselines=baselines, **attr_kwargs)
                    if positive_only:
                        at_map = at_map.clamp_min(0.0)

                    fn = f"{method}_{btype}_{cls}_{base}.png"
                    visualize_attr(at_map, img, os.path.join(out_dir, fn))

            # → alle anderen Methoden
            else:
                at_map = explainer.attribute(img, **attr_kwargs)
                if positive_only:
                    at_map = at_map.clamp_min(0.0)

                fn = f"{method}_{cls}_{base}.png"
                visualize_attr(at_map, img, os.path.join(out_dir, fn))

if __name__ == "__main__":
    main()
