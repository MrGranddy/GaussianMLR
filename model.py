import torch.nn as nn
import torchvision.models as models


class SqueezeNetBackbone(nn.Module):
    def __init__(self):
        super(SqueezeNetBackbone, self).__init__()
        self.model = models.squeezenet1_0(pretrained=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Model(nn.Module):
    def __init__(self, n_classes, feature_extractor="simple", pretrained=False):
        super(Model, self).__init__()

        self.n_classes = n_classes

        if feature_extractor == "simple":
            self.feature_extractor = SimpleBackbone()
            out_size = 256

        elif feature_extractor == "squeezenet":
            self.feature_extractor = SqueezeNetBackbone()
            out_size = 512

        elif feature_extractor == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            out_size = 512

        elif feature_extractor == "resnet50":
            self.feature_extractor = models.resnet50(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            out_size = 2048

        if pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        if feature_extractor == "simple":
            self.fc = nn.Sequential(
                nn.Linear(out_size, 128), nn.ReLU(), nn.Linear(128, self.n_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(out_size, 512), nn.ReLU(), nn.Linear(512, self.n_classes)
            )

    def forward(self, x):

        features = self.feature_extractor(x)
        logits = self.fc(features)

        return logits


class LSEPModel(nn.Module):
    def __init__(self, n_classes, feature_extractor="simple", pretrained=False):
        super(LSEPModel, self).__init__()

        self.n_classes = n_classes

        if feature_extractor == "simple":
            self.feature_extractor = SimpleBackbone()
            out_size = 256

        elif feature_extractor == "squeezenet":
            self.feature_extractor = SqueezeNetBackbone()
            out_size = 512

        elif feature_extractor == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            out_size = 512

        elif feature_extractor == "resnet50":
            self.feature_extractor = models.resnet50(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            out_size = 2048

        if pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        if feature_extractor == "simple":
            self.fc = nn.Sequential(
                nn.Linear(out_size, 128), nn.ReLU(), nn.Linear(128, self.n_classes)
            )
            self.threshold_fc = nn.Sequential(
                nn.Linear(out_size, 128), nn.ReLU(), nn.Linear(128, self.n_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(out_size, 512), nn.ReLU(), nn.Linear(512, self.n_classes)
            )
            self.threshold_fc = nn.Sequential(
                nn.Linear(out_size, 512), nn.ReLU(), nn.Linear(512, self.n_classes)
            )

    def forward(self, x):

        features = self.feature_extractor(x)

        scores = self.fc(features)
        thresholds = self.threshold_fc(features)

        return scores, thresholds


class GaussianModel(nn.Module):
    def __init__(self, n_classes, feature_extractor="simple", pretrained=False):
        super(GaussianModel, self).__init__()

        self.n_classes = n_classes

        if feature_extractor == "simple":
            self.feature_extractor = SimpleBackbone()
            out_size = 256

        elif feature_extractor == "squeezenet":
            self.feature_extractor = SqueezeNetBackbone()
            out_size = 512

        elif feature_extractor == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            out_size = 512

        elif feature_extractor == "resnet50":
            self.feature_extractor = models.resnet50(pretrained=pretrained)
            self.feature_extractor.fc = nn.Identity()
            out_size = 2048

        if pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        if feature_extractor == "simple":
            self.gauss_params_fc = nn.Sequential(
                nn.Linear(out_size, 128), nn.ReLU(), nn.Linear(128, 2 * self.n_classes)
            )
        else:
            self.gauss_params_fc = nn.Sequential(
                nn.Linear(out_size, 512), nn.ReLU(), nn.Linear(512, 2 * self.n_classes)
            )

    def forward(self, x):

        features = self.feature_extractor(x)
        gauss_params = self.gauss_params_fc(features)

        mean = gauss_params[:, : self.n_classes]
        logvar = gauss_params[:, self.n_classes :]

        return mean, logvar
