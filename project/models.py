import torch.nn as nn
from torchvision import models


class ClassifierModel(nn.Module):
    def __init__(self, backbone_name='vgg11', pretrained=True, num_classes=2):
        super().__init__()
        if backbone_name == 'vgg11':
            backbone = models.vgg11(pretrained=pretrained)
            self.features = backbone.features
            self.avgpool  = backbone.avgpool
            in_features   = backbone.classifier[-1].in_features
            classifier    = list(backbone.classifier.children())[:-1]
            classifier.append(nn.Linear(in_features, num_classes))
            self.classifier = nn.Sequential(*classifier)

        elif backbone_name in ['resnet18', 'resnet50']:
            # ResNet nutzen
            backbone = getattr(models, backbone_name)(pretrained=pretrained)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()   # entferne alten Kopf
            self.features  = backbone
            self.avgpool   = nn.Identity()  # nicht benötigt
            self.classifier = nn.Linear(in_features, num_classes)

        elif backbone_name.startswith('efficientnet'):
            backbone = getattr(models, backbone_name)(pretrained=pretrained)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            self.features  = backbone
            self.avgpool   = nn.Identity()
            self.classifier = nn.Linear(in_features, num_classes)

        elif backbone_name == 'densenet121':
            backbone = models.densenet121(pretrained=pretrained)
            # DenseNet hat backbone.features und backbone.classifier
            in_features = backbone.classifier.in_features
            # Entferne den alten Classifier
            backbone.classifier = nn.Identity()
            self.features = backbone.features
            # Global Pooling (DenseNet liefert bereits einen 1×1 Feature-Map nach features)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(in_features, num_classes)


        else:
            raise ValueError(f"Unbekannter Backbone: {backbone_name}")

    def forward(self, x):
        x = self.features(x)
        # falls Spatial-Pool nötig (nur für VGG)
        if hasattr(self, 'avgpool'):
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

