from pytorch_lightning import LightningModule

from torchmetrics import CohenKappa,Accuracy
from torch import optim, argmax, nn
from src.models.components.classifier import Classifier

class ClassiferFeatureSubset_Simple(LightningModule):
    def __init__(self,
                 lr,
                 weight_decay,
                 z_size):
        super().__init__()
        self.classifer = Classifier(z_size=z_size)
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = lr
        self.weight_decay = weight_decay

        #         self.val_kappa = CohenKappa(num_classes=3)
        #         self.test_kappa = CohenKappa(num_classes=3)
        self.val_kappa = Accuracy(num_classes=3)
        self.test_kappa = Accuracy(num_classes=3)

    def forward(self, z):
        return self.classifer(z)

    def training_step(self, batch, batch_idx):
        z, y = batch

        logits = self.forward(z)
        return {"logits": logits, "target": y}

    def training_step_end(self, outputs):
        logits = outputs['logits']
        y = outputs['target']
        preds = argmax(logits, dim=1)
        loss = self.loss(input=logits, target=y)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        z, y = batch
        logits = self.forward(z)

        return {"logits": logits, "target": y}

    def validation_step_end(self, outputs):
        logits = outputs['logits']
        y = outputs['target']
        preds = argmax(logits, dim=1)
        loss = self.loss(input=logits, target=y)

        self.val_kappa(preds, y)
        self.log('validation/kappa', self.val_kappa, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return {'logits': logits,
                'predictions': preds,
                'target': y}

    def test_step(self, batch, batch_idx):
        z, y = batch
        logits = self.forward(z)

        return {"logits": logits,
                "target": y}

    def test_step_end(self, outputs):
        logits = outputs['logits']
        y = outputs['target']
        preds = argmax(logits, dim=1)
        loss = self.loss(input=logits, target=y)

        self.test_kappa(preds, y)
        self.log('test/kappa', self.test_kappa, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return {'logits': logits,
                'predictions': preds,
                'target': y}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return optimizer