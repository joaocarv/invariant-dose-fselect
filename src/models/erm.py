from pytorch_lightning import LightningModule
import torchvision


from torchmetrics.functional import accuracy, f1, auc, cohen_kappa, matthews_corrcoef, precision, recall, specificity
from torchmetrics import AUC, Accuracy, MatthewsCorrCoef, CohenKappa, F1Score, Recall,Specificity, Precision, ConfusionMatrix
from src.models.components.backbone import Encoder
from torch import nn,cat,argmax,optim,no_grad
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import StepLR


class ERM(LightningModule):

    def __init__(self,
                 model,
                 lr,
                 weight_decay,
                 output_size,
                 pretrained=True,
                 fix_weights=None,
                 input_channels=3,
                 optimizer='adam',
                 momentum=None):
        super().__init__()

        self.save_hyperparameters()
        self.encoder = Encoder(model=model,
                                         pretrained=pretrained,
                                         input_channels=input_channels)

        self.classifier = nn.Linear(in_features=self.encoder.number_features,
                                    out_features=output_size)

        self.loss = nn.CrossEntropyLoss()
        self.num_classes = output_size
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.pretrained = pretrained
        self.fix_weights = fix_weights
        self.optimizer = optimizer
        self.momentum = momentum

        # Eval metrics
        self.train_acc = Accuracy(num_classes=self.num_classes,average='macro')
        
        self.train_mcc = MatthewsCorrCoef(num_classes=self.num_classes)
        self.train_kappa = CohenKappa(num_classes=self.num_classes)
        self.train_f1 = F1Score(num_classes=self.num_classes,average='macro')
        self.train_specificity = Specificity(num_classes=self.num_classes,average='macro')
        self.train_sensitivity = Recall(num_classes=self.num_classes,average='macro')
        self.train_precision = Precision(num_classes=self.num_classes,average='macro')

        
        self.val_acc = Accuracy(num_classes=self.num_classes,average='macro')
        
        self.val_mcc = MatthewsCorrCoef(num_classes=self.num_classes)
        self.val_kappa = CohenKappa(num_classes=self.num_classes)
        self.val_f1 = F1Score(num_classes=self.num_classes,average='macro')
        self.val_specificity = Specificity(num_classes=self.num_classes,average='macro')
        self.val_sensitivity = Recall(num_classes=self.num_classes,average='macro')
        self.val_precision = Precision(num_classes=self.num_classes,average='macro')

        self.test_mcc = MatthewsCorrCoef(num_classes=self.num_classes)
        self.test_kappa = CohenKappa(num_classes=self.num_classes)
        self.test_f1 = F1Score(num_classes=self.num_classes,average='macro')
        self.test_specificity = Specificity(num_classes=self.num_classes,average='macro')
        self.test_sensitivity = Recall(num_classes=self.num_classes,average='macro')
        self.test_precision = Precision(num_classes=self.num_classes,average='macro')

    def forward(self, x):
        if self.pretrained is True and self.fix_weights is not False:
            self.encoder.eval()
            with no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        return features,self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        _,logits = self.forward(x)


        # training metrics
        return {"logits": logits, "target": y}


    def training_step_end(self,outputs):

        logits = outputs['logits']
        y = outputs['target']
        

        preds = argmax(logits, dim=1)
        loss = self.loss(input=logits, target=y)
        self.log('train/class_loss', loss, prog_bar=False, on_step=True, on_epoch=False)


  
        self.train_acc(preds,y)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.train_auc(preds, y)
        # self.log('train/auc', self.train_auc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.train_f1(preds, y)
        self.log('train/f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.train_mcc(preds, y)
        self.log('train/mcc',self.train_mcc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.train_kappa(preds,y)
        self.log('train/kappa', self.train_kappa, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.train_sensitivity(preds, y)
        self.log('train/sensitivity', self.train_sensitivity, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.train_specificity(preds, y)
        self.log('train/specificity', self.train_specificity, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.train_precision(preds, y)
        self.log('train/precision', self.train_precision, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return {'loss': loss}



    def validation_step(self, batch, batch_idx):
        x, y = batch

        z,logits = self.forward(x)

        return {'logits': logits,
                'target': y,
                'repres': z,
                'x': x}

    def validation_step_end(self, outputs):

        x = outputs['x']
        logits = outputs['logits']
        y = outputs['target']
        z = outputs['repres']

        loss = self.loss(logits, y)
        preds = argmax(logits, dim=1)

        self.val_acc(preds,y)
        self.log('validation/acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('validation/class_loss', loss, prog_bar=False, on_step=False, on_epoch=True)

        self.val_f1(preds, y)
        self.log('validation/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.val_mcc(preds, y)
        self.log('validation/mcc', self.val_mcc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.val_kappa(preds, y)
        self.log('validation/kappa', self.val_kappa, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.val_sensitivity(preds, y)
        self.log('validation/sensitivity', self.val_sensitivity, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.val_specificity(preds, y)
        self.log('validation/specificity', self.val_specificity, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.val_precision(preds, y)
        self.log('validation/precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)


        return {'logits': logits,
                'predictions': preds,
                'target': y,
                'repres': z,
                'x': x}


    def test_step(self, batch, batch_idx, dataloader_idx=None):

        x, y = batch

        z,logits = self.forward(x)

        return {'logits': logits,
                'target': y,
                'repres': z,
                'x': x,
                'testloader_idx':dataloader_idx}
        


    def test_step_end(self, outputs):

        x = outputs['x']
        logits = outputs['logits']
        y = outputs['target']
        z = outputs['repres']

        testloader_idx=outputs['testloader_idx']
        if testloader_idx is not None:
            testloader_tag = 'test_id' if testloader_idx==0 else 'test_od'
        else:
            testloader_tag = 'test'
        preds = argmax(logits, dim=1)
        
        
        self.test_f1(preds, y)
        self.log('test/'+testloader_tag+'_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.test_mcc(preds, y)
        self.log('test/'+testloader_tag+'_mcc', self.test_mcc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.test_kappa(preds, y)
        self.log('test/'+testloader_tag+'_kappa', self.test_kappa, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.test_sensitivity(preds, y)
        self.log('test/'+testloader_tag+'_sensitivity', self.test_sensitivity, on_step=False, on_epoch=True, prog_bar=False,
                 logger=True)
        self.test_specificity(preds, y)
        self.log('test/'+testloader_tag+'_specificity', self.test_specificity, on_step=False, on_epoch=True, prog_bar=False,
                 logger=True)
        self.test_precision(preds, y)
        self.log('test/'+testloader_tag+'_precision', self.test_precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)


        return {'logits': logits,
                'predictions': preds,
                'target': y,
                'repres': z,
                'x': x,
                'testloader_tag':testloader_tag}


    def test_epoch_end(self, outputs):


        y = cat([output['target'] for output in outputs])
        y = y.to('cpu').numpy()
        z = cat([output['repres'] for output in outputs])
        z = z.to('cpu').numpy()


        df = pd.DataFrame(z,columns=[str(i) for i in range(z.shape[1])])
        df['target'] = [str(y) for y in y.tolist()]
        cols = df.columns.tolist()
        df = df[cols[-1:] + cols[:-1]]

        self.logger.log_table('test/representations',dataframe=df)




    def configure_optimizers(self):
        if self.optimizer == 'sdg':
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.learning_rate,
                                  momentum=self.momentum,
                                  weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = StepLR(optimizer,step_size=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler},
