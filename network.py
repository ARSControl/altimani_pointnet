# Import PyTorch Lightning

from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch, torch.nn as nn, torch.nn.functional as F
from pytorch_lightning import LightningModule
from utils import DEVICE
from typing import Any, Optional, Tuple
from termcolor import colored

from torchmetrics.classification import Accuracy

num_points = 750

class NeuralClassifier(LightningModule):

  """ Classifier Neural Network """

  def __init__(self, input_shape, output_shape, optimizer='Adam', lr=0.005, loss_function='cross_entropy'):

    super(NeuralClassifier, self).__init__()

    print(f'\nNeural Classifier\nInput Shape: {input_shape} | Output Shape: {output_shape}')
    print(f'Optimizer: {optimizer} | Learning Rate: {lr} | Loss Function: {loss_function}\n\n')

    # Compute Input and Output Sizes
    self.input_size, self.output_size = input_shape[0], output_shape[0]
    self.hidden_size = 512         ##########################################

    self.accuracy = Accuracy(task='multiclass', num_classes=self.output_size)

    print('Input Shape, Hidden Size, Output Shape NN:  ', self.input_size, self.hidden_size, self.output_size)

    # Create Fully Connected Layers
    self.net = nn.Sequential(
      nn.Linear(self.input_size, self.hidden_size),
      nn.ReLU(),
      # nn.Dropout(0.5),
      nn.Linear(self.hidden_size, self.hidden_size),
      nn.ReLU(),
      nn.Linear(self.hidden_size, self.output_size)
    ).to(DEVICE)

    # Instantiate Loss Function and Optimizer
    self.loss_function = getattr(torch.nn.functional, loss_function)
    self.optimizer     = getattr(torch.optim, optimizer)
    self.learning_rate = lr
    # print('\n DEBUG post convoluzioni lineari and ReLu \n')
    print('\nRete Creata')

  def forward(self, x:torch.Tensor) -> torch.Tensor:
   
    # Forward Pass through Fully Connected Layers
    out = self.net(x)

    # Softmax for Classification
    out = F.softmax(out, dim=1)
    
    return out
  
  def configure_optimizers(self):
    # Return Optimizer
    return self.optimizer(self.parameters(), lr = self.learning_rate)
  
  def compute_loss(self, batch:Tuple[torch.Tensor, torch.Tensor], log_name:str) -> torch.Tensor:

    # Get X,Y from Batch
    x, y = batch

    # Forward Pass
    y_pred = self(x)

    # Compute Loss
    loss = self.loss_function(y_pred, y.float())
    self.log(log_name, loss)

    return loss

  def training_step(self, batch, batch_idx):

    
    loss = self.compute_loss(batch, 'train_loss')
    return {'loss': loss}
  

  def test_step(self, batch, batch_idx):
    #####
    y_pred = self(batch[0])  # Assuming batch[0] contains input data
    accuracy = self.accuracy(y_pred, batch[1])
    self.log('test_accuracy', accuracy, on_step=False, on_epoch=True)
    ######
    loss = self.compute_loss(batch, 'test_loss')
    return {'test_loss': loss,'test_accuracy': accuracy}
