from torch import nn

class Classifier(nn.Module):
    def __init__(self,
                 z_size=500):
        super().__init__()

        self.classifier = nn.Sequential(nn.Linear(in_features=z_size, out_features=1000),
                                        nn.ReLU(),
                                        nn.Linear(in_features=1000,out_features=3)
                                        )


    def forward(self, z):

        return  self.classifier(z)
