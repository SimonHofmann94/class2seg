# class2seg

## Projektstatus

Wir haben eine vollständige Klassifikations-Pipeline für den BSData-Datensatz implementiert und getestet.

## Änderungsübersicht (XAI-Integration)

- Multi-Class Support: Einführung von target_classes statt einzelner target_class in der Config.

- Baselines für DeepLift & DeepLiftShap: Automatisches Durchprobieren von drei Baseline-Typen (zero, gaussian, dataset_mean).

- Positive Attributions: Negative Attributions werden per clamp_min(0) auf 0 gesetzt.

- Dateinamenskonvention: Outputs heißen nun <method>_<baseline?>_<class>_<orig_basename>.png.

- Dataset-Rückgabe: datasets.py liefert jetzt (img, label, base_name) für eindeutige Benennung.

## Datenhandling (project/datasets.py)

- Einlesen von BSData mit JSON-Annotationen.

- 3-fach-Split: Train (80%), Val (10%), Test (10%).

- Rückgabe von Bild, Label und Basis-Dateiname für XAI-Outputs.

## Modelle (project/models.py)

### Unterstützte Backbones:

- VGG11, VGG16

- ResNet18, ResNet50

- EfficientNet-B0

- DenseNet121

## Training (project/train.py)

- Oversampling via WeightedRandomSampler.

- Class-Weights in CrossEntropyLoss.

- Starke Data-Augmentation (RandomResizedCrop, Flips, Rotation, Perspective, ColorJitter, GaussianBlur, RandomErasing).

- Input-Normalisierung für ImageNet-Backbones.

- EarlyStopping und ReduceLROnPlateau Scheduler.

## Evaluation (project/evaluation.py)

- Metriken: Accuracy, Precision, Recall, F1, ROC AUC.

- Konfusionsmatrix.

- unabhängiges Test-Set.

- Bisherige Ergebnisse

## ResNet50 auf Test-Set

Accuracy : 0.7909
Precision: 0.6406
Recall   : 1.0000
F1 Score : 0.7810
ROC AUC  : 0.9975
Confusion Matrix:
[[46 23]
 [ 0 41]]

DenseNet121 auf Test-Set

Accuracy : 0.9818
Precision: 0.9535
Recall   : 1.0000
F1 Score : 0.9762
ROC AUC  : 0.9975
Confusion Matrix:
[[67  2]
 [ 0 41]]

Nächste Schritte

Threshold-Tuning (Precision-Recall-Kurve, F1-Optimierung).

Visuelle Inspektion der False Positives.

XAI-Attributionskarten (alle Methoden) & Postprocessing (Thresholding, Morphologie).

Generierung und Evaluation von Segmentierungsmasken.

Installation

# Repo klonen
git clone git@github.com:SimonHofmann94/class2seg.git
cd class2seg

# virtuelle Umgebung erstellen
python3 -m venv venv
source venv/bin/activate

# Projekt installierbar machen
pip install -e .
# Dependencies installieren
pip install -r requirements.txt

Konfiguration

Alle Hyperparameter und Pfade werden über Hydra-Configs gesteuert (config/).

config/data/default.yaml: Pfade, train_val_split, test_split

config/model/default.yaml: backbone, pretrained, num_classes

config/train/default.yaml: batch_size, lr, optimizer, EarlyStopping, etc.

config/augment/default.yaml: Data-Augmentation-Flags

config/output/default.yaml: Ausgabepfade für Attributions, Masken, Visuals

config/xai/default.yaml: XAI-Methode, target_classes, baselines, positive_only, max_samples

CLI-Befehle

Training

# ResNet50
python -m project.train model.backbone=resnet50 model.num_classes=2

# DenseNet121
python -m project.train model.backbone=densenet121 model.num_classes=2

Evaluation (Test-Set)

# ResNet50
python -m project.evaluation model.backbone=resnet50 model.num_classes=2

# DenseNet121
python -m project.evaluation model.backbone=densenet121 model.num_classes=2

XAI-Attributions

# Integrated Gradients
python project/xai.py \
  model.backbone=densenet121 \
  xai.method=integrated_gradients \
  xai.target_classes=[1] \
  xai.baseline_types=[]

# Saliency
python project/xai.py \
  model.backbone=densenet121 \
  xai.method=saliency \
  xai.target_classes=[1]

# DeepLift mit allen Baselines
python project/xai.py \
  model.backbone=densenet121 \
  xai.method=deeplift \
  xai.target_classes=[1] \
  xai.baseline_types=[zero,gaussian,dataset_mean]

# GradCam++
python project/xai.py \
  model.backbone=densenet121 \
  xai.method=gradcam++ \
  xai.target_classes=[1] \
  xai.baseline_types=[]

