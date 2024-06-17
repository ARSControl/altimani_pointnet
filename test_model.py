# Import  
import torch, trimesh, os
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from utils import StartTestingCallback, StartTrainingCallback
from utils import FOLDER, FOLDER_MODEL, PATH, save_model, DEVICE
from algoritm import MyAlgo, load_model_from_file
import matplotlib.pyplot as plt
from trimesh import Trimesh

# Set Torch Matmul Precision
torch.set_float32_matmul_precision('high')

# Define training hyperparameters
min_epochs, max_epochs = 150, 200   ##mod
min_delta, patience = 0, 50
fast_dev_run = False

batch_size = 64  ##mod 
#  modelnet10:  77 -> acc=0,856387   ............75->0,857268......73->0,85947.... 71->0,85969... 72->0,85275  .......70->0,839/0.8592
optimizer, learning_rate, loss_function = 'Adam', 0.001, 'cross_entropy'   ##mod

#Inside utils.py you can find:
#num_points = 1124     #number of point that are use for the elaboration
#num_classes = 10      #number of classes of objest
accuracy_values = []

MODEL_PATH = r'/home/davide/ROS/altimani_ws/3Dpc - v3/model'

def main():

  # Create Classification Training Algorithm
  algo = MyAlgo(batch_size, optimizer, learning_rate, loss_function)
  print('\nMAIN: Algoritmo Creato')
  
  # Load the model if it exists, otherwise create a new on
  if os.path.exists(f'{MODEL_PATH}/model.pth'):

    with open(f'{MODEL_PATH}/model.pth', 'rb') as FILE:

        # Get Model
        model = algo.getModel()

        # Load Model Weights
        checkpoint = torch.load(FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

    print("\nModel exist. Loading done.")

  else: raise Exception("\nModel does not exist.")

  ####################################### put on video acloudpoint
  # mesh = trimesh.load(os.path.join(PATH, "chair/train/chair_0007.off"))
  # mesh = trimesh.load(os.path.join(PATH, "airplane/train/airplane_0001.off"))
  # mesh = trimesh.load(os.path.join(PATH, "airplane/train/airplane_0010.off"))
  # mesh:Trimesh = trimesh.load(os.path.join(PATH, "door/train/door_0010.off"))
  mesh:Trimesh = trimesh.load(os.path.join(PATH, "car/train/car_0010.off"))
  mesh:Trimesh = trimesh.load(os.path.join(PATH, "guitar/train/guitar_0010.off"))
  points = mesh.sample(1024)
  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(111, projection="3d")
  ax.scatter(points[:, 0], points[:, 1], points[:, 2])
  ax.set_axis_off()
  plt.show()
  ##########################################################
  # Flatten dei punti

  points_flat = points.reshape(-1)  # shape: (num_point * 3,)
  
  print('\nDEBUG POINT flat:', points_flat)
  
  # Converti i punti in tensore e aggiungi una dimensione per il batch
  points_tensor = torch.tensor(points_flat, dtype=torch.float32).unsqueeze(0)
  
  print("Dimensioni dei tensori prima del passaggio al modello:")

  print("Punti tensor:", points_tensor.shape)


  # Flatten dei punti
  points_flat = points.flatten()  # shape: (1024 * 3,)
  
  print('\nDEBUG POINT flat:', points_flat)

  # Converti i punti in tensore e aggiungi una dimensione per il batch
  points_tensor = torch.tensor(points_flat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

  print("Dimensioni dei tensori prima del passaggio al modello:")
  print("Punti tensor:", points_tensor.shape)

  # Passa i punti attraverso il modello per ottenere le previsioni
  with torch.no_grad():
    predictions = model(points_tensor)
    predicted_label = torch.argmax(predictions, dim=1).item()
    # print("Etichetta prevista:", predicted_label)
    print("Previsioni: ", predictions)
    print("Etichetta prevista: ", predicted_label)
    print("Dimensioni delle previsioni: ", predictions.shape)


  ###############################


  print('\n   END MAIN')


if __name__ == '__main__':
  main()