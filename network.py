import torch, torch.nn as nn, torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import LightningModule
from utils import DEVICE
from typing import Any, Optional, Tuple
from termcolor import colored

from torchmetrics.classification import Accuracy


from torch.optim.lr_scheduler import StepLR

num_points = 2000     #number of point that are use for the elaboration
num_classes = 10      #number of classes of objest

class NeuralClassifier(LightningModule):
    def __init__(self, input_shape, output_shape, optimizer='Adam', lr=0.001, loss_function='cross_entropy'):
        super(NeuralClassifier, self).__init__()
        self.input_size, self.output_size = input_shape[-1], output_shape[-1]
        self.hidden_size = 512
        self.accuracy = Accuracy(task='multiclass', num_classes=10)
        # self.accuracy = Accuracy(task='multiclass', num_classes=self.output_size)
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )
        self.loss_function = getattr(torch.nn.functional, loss_function)
        self.learning_rate = lr
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x):
        out = self.net(x)
        return F.softmax(out, dim=1)
    
    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss'
            }
        }
    def compute_loss(self, batch:Tuple[torch.Tensor, torch.Tensor], log_name:str) -> torch.Tensor:
        # Get X,Y from Batch
        x, y = batch
        y_pred = self(x)
        # Compute Loss
        loss = self.loss_function(y_pred, y.float())
        self.log(log_name, loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, 'train_loss')
        return {'loss': loss}


    def test_step(self, batch, batch_idx):
        y_pred = self(batch[0])  
        # accuracy : lable predict/ lable of the pointcloud ??
        accuracy = self.accuracy(y_pred, batch[1])
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True)
        loss = self.compute_loss(batch, 'test_loss')
        return {'test_loss': loss,'test_accuracy': accuracy,'pred': y_pred,'target': batch[1]}
