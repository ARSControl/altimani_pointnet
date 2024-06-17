import os, torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from termcolor import colored

# Get Torch Device ('cuda' or 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Project Folder (ROOT Project Location)
FOLDER = os.path.abspath(os.path.join(os.path.dirname('/home/davide/ROS/altimani_ws/3Dpc/model/model.pth')))
PATH = "/home/davide/ROS/altimani_ws/3Dpc/ModelNet10"

def save_model(path:str, file_name:str, model:LightningModule):

  """ Save File Function """
  # Create Directory if it Doesn't Exist
  os.makedirs(path, exist_ok=True)
  # counter = 1

  # # Check if the File Already Exists
  # while os.path.exists(os.path.join(path, file_name)):
  #   # Get the File Name and Extension
  #   file_name, file_extension = os.path.splitext(file_name)
  #   # Append a Number to the File Name to Make it Unique
  #   counter += 1
  #   file_name = f'{file_name}_{counter}{file_extension}'

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
