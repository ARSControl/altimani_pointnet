# Import  
import torch, trimesh, os
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from utils import StartTestingCallback, StartTrainingCallback
from utils import FOLDER, FOLDER_MODEL, PATH, save_model
from algoritm import MyAlgo, load_model_from_file
import matplotlib.pyplot as plt

# Set Torch Matmul Precision
torch.set_float32_matmul_precision('high')

# Define training hyperparameters
min_epochs, max_epochs = 22000, 800000   ##mod
min_delta, patience = 0, 200
fast_dev_run = False

batch_size = 512       ##mod
optimizer, learning_rate, loss_function = 'Adam', 1e-3, 'cross_entropy'   ##mod

#Inside utils.py you can find:
#num_points = 1124     #number of point that are use for the elaboration
#num_classes = 10      #number of classes of objest

MODEL_PATH = r'/home/davide/ROS/altimani_ws/3Dpc - v2/model'

def main():

  # Create Classification Training Algorithm
  algo = MyAlgo(batch_size, optimizer, learning_rate, loss_function)
  print('\nMAIN: Algoritmo Creato')
  
  # Load the model if it exists, otherwise create a new on
  if os.path.exists(f'{MODEL_PATH}/model.pth'):
    model = load_model_from_file(f'{MODEL_PATH}', 'model.pth', batch_size)
    print("\nModel exist. Loading done.")
  else:
    print("\nModel does not exist. Create new one.")
    model = algo.getModel()
    # model_path = FOLDER
  print("\nMAIN: Model Loaded")

  # Prepare Dataset
  train_dataloader, test_dataloader = algo.getDataloaders()
  print("\nMAIN: Dataloader Loaded")

  # Create Trainer Module
  trainer = Trainer(

    # Devices
    devices= 'auto',
    # Hyperparameters
    # min_epochs = min_epochs,
    max_epochs = max_epochs,
    log_every_n_steps = 1,
    # Instantiate Early Stopping Callback
    callbacks = [StartTrainingCallback(), StartTestingCallback(),
                 EarlyStopping(monitor='train_loss', mode='min', min_delta=min_delta, patience=patience, verbose=True)
                 ],
    # Custom TensorBoard Logger
    logger = pl_loggers.TensorBoardLogger(save_dir=f'{FOLDER}/model/data/logs/'),
    # Developer Test Mode
    fast_dev_run = fast_dev_run
  )

  print('\nMAIN: Start Training\n')
  # Start Training
  trainer.fit(model, train_dataloader)

  print('\nMAIN: End Training, Start Test\n')

  # Start Testing
  test_results =  trainer.test(model, test_dataloader)
  print("\nMAIN: End Test")
  
  # Save Model
  # save_model(os.path.join(FOLDER_MODEL, 'model'), 'model.pth', model)

  save_model(os.path.join(FOLDER_MODEL), 'model.pth', model)
  
 
  #DEBUG
  print("\nMAIN: Visualization")
  print('\nTest results:', test_results)
  print('\n   END MAIN')


if __name__ == '__main__':
  main()