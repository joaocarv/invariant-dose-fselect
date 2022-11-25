import torchvision
from torch import nn

class Encoder(nn.Module):
    def __init__(self, model="resnet50", pretrained=True, input_channels=3):
        super(Encoder, self).__init__()


        if 'resnet' in model:
            if str(18) in model:
                self.feature_extractor = torchvision.models.resnet18(True)
            elif str(34) in model:
                self.feature_extractor = torchvision.models.resnet34(True)
            elif str(50) in model:
                self.feature_extractor = torchvision.models.resnet50(True)
            elif str(101) in model:
                self.feature_extractor = torchvision.models.resnet101(True)

            if input_channels != 3:
                self.feature_extractor.conv1 = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False
                )
                self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.number_features = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()

        elif 'vit' in model:
            self.feature_extractor = torchvision.model.vit_l_32()

            self.number_features = self.feature_extractor.heads.head.in_features


    def forward(self, x):
        h = self.feature_extractor(x)

        return h