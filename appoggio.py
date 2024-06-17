io ho questi 5 moduli
1) main.py :
[
# Import  
import torch, trimesh, os
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from utils import StartTestingCallback, StartTrainingCallback
from utils import FOLDER, PATH, save_model
from algoritm import MyAlgo
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')

min_epochs, max_epochs = 100, 200   
min_delta, patience = 0, 50
fast_dev_run = False
batch_size = 64
optimizer, learning_rate, loss_function = 'Adam', 1e-3, 'cross_entropy'

def main():

  # Create Classification Training
  algo = MyAlgo(batch_size, optimizer, learning_rate, loss_function)

  print('\nMAIN: Algoritmo Creato')
  model = algo.getModel()
  model_path = FOLDER
   
  print("\nMAIN: Modello NN Caricato")

  train_dataloader, test_dataloader = algo.getDataloaders()

  print("\nMAIN: Dataloader Caricati")

  # Create Trainer Module
  trainer = Trainer(
    # Devices
    devices = 'auto', 
    accelerator = 'auto',
    # Hyperparameters
    min_epochs = min_epochs,
    max_epochs = max_epochs,
    log_every_n_steps = 1,
    # Instantiate Early Stopping Callback
    callbacks = [StartTrainingCallback(), StartTestingCallback(),
        EarlyStopping(monitor='train_loss', mode='min', min_delta=min_delta, patience=patience, verbose=True)
        ],
    # Custom TensorBoard Logger
    logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/data/logs/'),
    # Developer Test Mode
    fast_dev_run = fast_dev_run
  )

  print('\nMAIN: Start Training\n')
  # Start Training
  trainer.fit(model, train_dataloaders=train_dataloader)
  print('\nMAIN: End Training, Start Test\n')
  trainer.test(model, dataloaders=test_dataloader)
  print("\nMAIN: End Test")
  # Save Model
  save_model('/home/fra/CodeFolderS/modelli', 'model.pth', model)
  print('\n   TREMINE MAIN')


if __name__ == '__main__':
  main()

]

2) network.py :
[
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch, torch.nn as nn, torch.nn.functional as F
from pytorch_lightning import LightningModule
from utils import DEVICE
from typing import Any, Optional, Tuple
from termcolor import colored

num_points = 750

class NeuralClassifier(LightningModule):

  """ Classifier Neural Network """

  def __init__(self, input_shape, output_shape, optimizer='Adam', lr=0.005, loss_function='cross_entropy'):
    super(NeuralClassifier, self).__init__()
    print(f'\nNeural Classifier\nInput Shape: {input_shape} | Output Shape: {output_shape}')
    print(f'Optimizer: {optimizer} | Learning Rate: {lr} | Loss Function: {loss_function}\n\n')
    # Compute Input and Output Sizes
    self.input_size, self.output_size = input_shape[0], output_shape[0]
    self.hidden_size = 512        
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
    loss = self.compute_loss(batch, 'test_loss')
    return {'test_loss': loss}

]

3) algoritm.py : 
[
import os, torch, trimesh
import numpy as np
from typing import Tuple
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from dataset import MyDataset
from network import NeuralClassifier, num_points
from utils import FOLDER, PATH, DEVICE
from numba import njit

class MyAlgo:

  """ 3D Training Class """

  def __init__(self, batch_size, optimizer, learning_rate, loss_function):
    # Get Database and Model Path
    self.model_path = os.path.join(FOLDER, 'model')
    # Prepare Dataloaders
    dataset_shapes = self.prepareDataloaders(batch_size)
    # Create Model
    self.createModel(dataset_shapes, optimizer, learning_rate, loss_function)

  def prepareDataloaders(self, batch_size: int) -> Tuple[torch.Size, torch.Size]:
    """ Prepare Dataloaders """
    datacare = self.parse_dataset(num_points)
    # sequences, labels = self.parse_dataset(num_points) 
    print("dentro dataloaders")
    sequences, labels, _ = datacare
    print('sembra OK')
    dataset= MyDataset(sequences, labels)
    assert sequences.shape[0] == labels.shape[0], 'Sequences and Labels must have the same length'
    assert torch.Size(sequences.reshape(sequences.shape[0], -1).shape) == dataset.getInputShape(), 'Dataset Input Shape must be equal to Sequences Shape'
    assert labels.shape[0] == dataset.getOutputShape()[0], 'Dataset Output Shape must be equal to Labels Shape'
    self.train_dataloader = DataLoader(dataset, batch_size=batch_size)
    self.test_dataloader  = DataLoader(dataset, batch_size=batch_size)

    return dataset.getInputShape(), dataset.getOutputShape()

  def getDataloaders(self) -> Tuple[DataLoader, DataLoader]:
    """ Get Dataloaders """
    return self.train_dataloader, self.test_dataloader

  def createModel(self, dataset_shape: Tuple[torch.Size, torch.Size], optimizer='SDG', lr=0.0005, loss_function='cross_entropy'):
    # Get Input and Output Sizes from Dataset Shapes
    input_size, output_size = torch.Size(list(dataset_shape[0])[1:]), torch.Size(list(dataset_shape[1])[1:])
    print(f'\nCreate Model:\nInput Shape: {dataset_shape[0]} | Output Shape: {dataset_shape[1]}')
    print(f'Input Size: {input_size} | Output Size: {output_size}')
    # Create NeuralNetwork Model
    self.model = NeuralClassifier(input_size, output_size, optimizer, lr, loss_function)
    self.model.to(DEVICE)

  def getModel(self) -> Tuple[LightningModule, str]:
    """ Get NN Model and Model Path """
    return self.model        #, self.model_path

  def parse_dataset(self, num_points) -> Tuple[np.ndarray, np.ndarray, dict]:
    print(os.listdir(PATH))
    folders = [dir for dir in sorted(os.listdir(PATH)) if os.path.isdir(os.path.join(PATH, dir))]
    class_map = {}
    points, labels = [], []
    for i, folder in enumerate(folders):
        print(f"Processing class: {os.path.basename(folder)}")
        class_map[i] = folder.split("/")[-1]
        train_files = os.path.join(f'{PATH}/{folder}', "train/")
        test_files = os.path.join(f'{PATH}/{folder}', "test/")
        points_train, labels_train = self.load_files(i, train_files, num_points)
        points_test, labels_test = self.load_files(i, test_files, num_points)
        points.extend(points_train)
        labels.extend(labels_train)
    return np.array(points), np.array(labels), class_map

  def load_files(self, i, T_files, num_points):
    # points, labels = np.asarray([]), np.asarray([])
    points, labels = [],[]
    for f in os.listdir(T_files):
        
        points.append(trimesh.load(f"{T_files}/{f}").sample(num_points))
        labels.append(i)
    return points, labels 
]

4) dataset.py :
[
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Union, Tuple
from utils import DEVICE
class MyDataset(Dataset):

  """ Dataset Class """

  def __init__(self, x:Union[torch.Tensor, np.ndarray], y:Union[torch.Tensor, np.ndarray]):
    self.x = x if torch.is_tensor(x) else torch.from_numpy(x).float()
    self.y = y if torch.is_tensor(y) else torch.from_numpy(y)
    print('\nDataset Creation, Original Inputs:')
    print(f'Input Size: {self.x.shape} | Output Size: {self.y.shape}')
    # Flattening X Input Data
    self.x = self.x.view(self.x.size(0), -1)
    print('Dataset Creation, Flattened Inputs:')
    print(f'Input Size: {self.x.shape} | Output Size: {self.y.shape}')
    if len(self.y.shape) == 1:  # Check if y is not one-hot encoded
      self.y = torch.nn.functional.one_hot(self.y)
    self.x = self.x.to(DEVICE)
    self.y = self.y.to(DEVICE)
    print(f'\nDataset Creation:\nInput Shape: {x.shape} | Output Shape: {y.shape}')
    print(f'Input Size: {self.x.shape} | Output Size: {self.y.shape}')

  def getInputShape(self) -> torch.Size:
    return self.x.shape

  def getOutputShape(self) -> torch.Size:
    return self.y.shape

  def __len__(self) -> int:
    return len(self.y)

  def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
    return self.x[idx], self.y[idx]

]

5) utils.py : 
[
import os, torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from termcolor import colored

# Get Torch Device ('cuda' or 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Project Folder (ROOT Project Location)
FOLDER = os.path.abspath(os.path.join(os.path.dirname('/home/fra/CodeFolderS/model/model.pth')))

PATH = "/home/fra/CodeFolderS/ModelNet10"

def save_model(path:str, file_name:str, model:LightningModule):

  """ Save File Function """

  # Create Directory if it Doesn't Exist
  os.makedirs(path, exist_ok=True)
  with open(os.path.join(path, file_name), 'wb') as FILE: torch.save(model.state_dict(), FILE)
  print(colored('\n\nModel Saved Correctly\n\n', 'green'))

# Print Start Training Info Callback
class StartTrainingCallback(Callback):
  # On Start Training
  def on_train_start(self, trainer, pl_module):
    print(colored('\n\nStart Training Process\n\n','yellow'))
  # On End Training
  def on_train_end(self, trainer, pl_module):
    print(colored('\n\nTraining Done\n\n','yellow'))

# Print Start Testing Info Callback
class StartTestingCallback(Callback):
  # On Start Testing
  def on_test_start(self, trainer, pl_module):
    print(colored('\n\nStart Testing Process\n\n','yellow'))
  # On End Testing
  def on_test_end(self, trainer, pl_module):
    print(colored('\n\n\nTesting Done\n\n','yellow'))

]