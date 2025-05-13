import torch.nn as nn
from torchvision import models


class ClassifierModel(nn.Module):
    """
    Wrapper für Klassifikations-Backbones (VGG11, VGG16 etc.) mit austauschbarem Classifier-Head.
    """
    def __init__(self,
                 backbone_name: str = 'vgg11',
                 pretrained: bool = True,
                 num_classes: int = 2):
        super().__init__()
        # Backbone auswählen
        if backbone_name == 'vgg11':
            backbone = models.vgg11(pretrained=pretrained)
        elif backbone_name == 'vgg16':
            backbone = models.vgg16(pretrained=pretrained)
        else:
            raise ValueError(f"Unbekannter Backbone: {backbone_name}")

        # Feature-Extraktor
        self.features = backbone.features

        # Classifier-Head anpassen
        # Standard-Classifier ist nn.Sequential, letzter Layer ist ein Linear
        orig_classifier = backbone.classifier
        in_features = orig_classifier[-1].in_features
        # Neuen Classifier bauen: alle alten Layers bis zum vorletzten, letzter Linear an neue num_classes
        classifier_layers = list(orig_classifier.children())[:-1]
        classifier_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        # Flatten vom Feature-Map-Tensor
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
