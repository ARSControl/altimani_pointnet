import os, torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from termcolor import colored

# Get Torch Device ('cuda' or 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Project Folder (ROOT Project Location) NEED TO CHANGE "/home/fra" with correct path of the folder
FOLDER = os.path.abspath(os.path.join(os.path.dirname('/home/davide/ROS/altimani_ws/3Dpc - v3')))
FOLDER_MODEL = os.path.abspath(os.path.join(os.path.dirname('/home/davide/ROS/altimani_ws/3Dpc - v3/model/model.pth')))

# PATH of the 3D pointcloud dataset saved ad "namefil.off" -> matrix[ num_points x 3 ] 
#root path :
# "/3Dpc/ModelNet10{
#                   bathtub[
#                           train(
#                                 file1.off,
#                                 file2.off,
#                                 ...),
#                           test(
#                                fileX1.off,
#                                fileX2.off, 
#                                 ...)
#                           ],
#                   bed[
#                       train(...),
#                       test(...)
#                       ],
#                   chair[...],
#                   ...}"
#
# PATH = "/home/davide/ROS/altimani_ws/3Dpc - v3/ModelNet10"
PATH = "/home/davide/ROS/altimani_ws/3Dpc - v3/ModelNet40"

def save_model(path:str, file_name:str, model:LightningModule):

  """ Save File Function """

  # Create Directory if it Doesn't Exist
  os.makedirs(path, exist_ok=True)
  # Need input of the model
  input_shape = model.input_size
  output_shape = model.output_size

  checkpoint = {
      'input_shape': input_shape,
      'output_shape': output_shape,
      'model_state_dict': model.state_dict(),
  }

  with open(os.path.join(path, file_name), 'wb') as FILE:
    torch.save(checkpoint, FILE)
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
