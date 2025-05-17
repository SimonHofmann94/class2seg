import os
import glob
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture
from skimage.morphology import opening, closing, disk, square, remove_small_objects


# ================================================
# Postprocessing für Attributionskarten → Segmentierungs-Masken
# ================================================

def binarize_threshold(attr_map: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """
    Binarisierung via Min-Max-Normierung und festem Schwellenwert, Otsu oder Perzentil.
    """
    # Normiere auf [0,1]
    attr = attr_map - attr_map.min()
    maxv = attr.max()
    if maxv > 0:
        attr = attr / maxv
    else:
        return np.zeros_like(attr, dtype=bool)

    method = cfg.postprocessing.threshold.method
    if method == "fixed":
        thr = cfg.postprocessing.threshold.value
    elif method == "otsu":
        thr = threshold_otsu(attr)
    elif method == "percentile":
        p = cfg.postprocessing.threshold.percentile
        thr = np.percentile(attr, p)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return attr > thr


def binarize_gmm(attr_map: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """
    Binarisierung via Gaussian Mixture Model.
    """
    # 1) Min-Max-Normierung auf [0,1]
    attr = attr_map - attr_map.min()
    maxv = attr.max()
    if maxv > 0:
        attr = attr / maxv
    else:
        # alles Null → leere Maske
        return np.zeros_like(attr, dtype=bool)
    # Flatten für GMM
    flat = attr.reshape(-1, 1)
    gmm_cfg = cfg.postprocessing.gmm
    gmm = GaussianMixture(
        n_components=gmm_cfg.n_components,
        covariance_type=gmm_cfg.covariance_type,
        random_state=cfg.train.seed
    )
    gmm.fit(flat)
    # harte Cluster-Zuordnung
    labels = gmm.predict(flat)
    # oder: posterior probabilities
    probs = gmm.predict_proba(flat)

    means = gmm.means_.flatten()
    fg_label = int(np.argmax(means))
    p_thr = cfg.postprocessing.gmm.get('probability_threshold', None)

    if p_thr is not None:
        # nur Pixel, die mit hoher Sicherheit in FG-Cluster liegen
        fg_probs = probs[:, fg_label]
        mask = (fg_probs.reshape(attr_map.shape) > p_thr)
    else:
        mask = (labels.reshape(attr_map.shape) == fg_label)
    return mask


def apply_morphology(mask: np.ndarray, cfg: DictConfig) -> np.ndarray:
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D for morphology, got shape {mask.shape}")
    mask = mask.astype(bool)
    morph = cfg.postprocessing.morphology

    if morph.pipeline == 'forest':
        # 1) Erster Closing
        se1 = disk(morph.closing1_size) if morph.selem_shape=='disk' else square(morph.closing1_size)
        mask = closing(mask, se1)
        # 2) Area-Opening: remove small objects
        mask = remove_small_objects(mask, min_size=morph.area_opening_min_size)
        # 3) Zweiter Closing
        se2 = disk(morph.closing2_size) if morph.selem_shape=='disk' else square(morph.closing2_size)
        mask = closing(mask, se2)
    else:
        # klassische Pipeline
        if morph.apply_opening:
            se = disk(morph.opening_size) if morph.selem_shape=='disk' else square(morph.opening_size)
            mask = opening(mask, se)
        if morph.apply_closing:
            se = disk(morph.closing_size) if morph.selem_shape=='disk' else square(morph.closing_size)
            mask = closing(mask, se)

    return mask


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Lädt Attributionskarten (.npy) ein, wendet Postprocessing an
    (Threshold/GMM + Morphologie) und speichert Binärmasken (.png).
    """
    root = hydra.utils.get_original_cwd()
    # Dynamisches Output-Verzeichnis basierend auf Postprocessing-Einstellungen
    pp = cfg.postprocessing
    if cfg.postprocessing.method == 'threshold':
        tcfg = pp.threshold
        if tcfg.method == 'fixed':
            subfolder = f"threshold_fixed_{tcfg.value}"
        elif tcfg.method == 'otsu':
            subfolder = "threshold_otsu"
        elif tcfg.method == 'percentile':
            subfolder = f"threshold_percentile_{tcfg.percentile}"
        else:
            subfolder = "threshold_unknown"
    elif cfg.postprocessing.method == 'gmm':
        gcfg = pp.gmm
        subfolder = f"gmm_{gcfg.n_components}_{gcfg.covariance_type}"
    else:
        subfolder = cfg.postprocessing.method

    in_dir = os.path.join(
        root,
        cfg.output.attribution_dir,
        cfg.model.backbone,
        cfg.xai.method
    )
    out_dir = os.path.join(
        root,
        cfg.output.mask_dir,
        cfg.model.backbone,
        cfg.xai.method,
        subfolder
    )
    os.makedirs(out_dir, exist_ok=True)

    # Alle .npy Attributionskarten verarbeiten
    for npy_path in glob.glob(os.path.join(in_dir, "*.npy")):
        base = os.path.splitext(os.path.basename(npy_path))[0]
        # ── Sanitize 'base', falls es als Tuple-String repräsentiert wurde:
        # z.B. "('01_..._crop_2',)" → "01_..._crop_2"
        if base.startswith("('") and base.endswith("',)"):
            base = base[2:-3]
        # Optional: weitergehendes Cleanup, falls nötig
        base = base.strip("()'\"")
        attr_map = np.load(npy_path)

        # Binarisierung
        if cfg.postprocessing.method == 'threshold':
            mask = binarize_threshold(attr_map, cfg)
        elif cfg.postprocessing.method == 'gmm':
            mask = binarize_gmm(attr_map, cfg)
        else:
            raise ValueError(f"Unknown postprocessing method: {cfg.postprocessing.method}")

        # Morphologische Nachbearbeitung
        mask = apply_morphology(mask, cfg)

        # Datei-Schema: <method>_<class>_<base>_mask.png
        cls = cfg.xai.target_classes[0] if cfg.xai.target_classes else cfg.xai.target_class
        fname = f"{cfg.postprocessing.method}_{cls}_{base}_mask.png"
        # Speichern als PNG (0=schwarz,255=weiß)
        arr = (mask.astype(np.uint8) * 255)
        img = Image.fromarray(arr, mode='L')
        img.save(os.path.join(out_dir, fname))


if __name__ == "__main__":
    main()
