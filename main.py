# Import  
import torch, trimesh, os
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from utils import StartTestingCallback, StartTrainingCallback
from utils import FOLDER, PATH, save_model


from algoritm import MyAlgo
import matplotlib.pyplot as plt

# Set Torch Matmul Precision
torch.set_float32_matmul_precision('high')

# fast_dev_run = True     #####mettere falsa per farre tutte le epoch
fast_dev_run = False


#######################################################################mod: = 64, 0.8 , 0.2
batch_size = 512
# train_set_size, test_set_size = 1, 1
optimizer, learning_rate, loss_function = 'AdamW', 1e-3, 'cross_entropy'

def main():

  # Create Classification Training
  algo = MyAlgo(batch_size, optimizer, learning_rate, loss_function)

  print('\nMAIN: Algoritmo Creato')
  model = algo.getModel()
  model_path = FOLDER
 
  ####################################### put on video acloudpoint
  # mesh = trimesh.load(os.path.join(PATH, "chair/train/chair_0007.off"))
  # points = mesh.sample(1024)
  # fig = plt.figure(figsize=(5, 5))
  # ax = fig.add_subplot(111, projection="3d")
  # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
  # ax.set_axis_off()
  # plt.show()
   ###############################
  
  print("\nMAIN: Modello NN Caricato")

  # Prepare Dataset
  train_dataloader, test_dataloader = algo.getDataloaders()

  # print("dataloader train structure:", type(train_dataloader))
  # print("dataloader train contents:", train_dataloader)
  # print("dataloader test structure:", type(test_dataloader))
  # print("dataloader test contents:", test_dataloader)

  print("\nMAIN: Dataloader Caricati")

  # Create Trainer Module
  trainer = Trainer(

    # Devices
    devices = 'auto', 
    accelerator = 'auto',

    # Hyperparameters
    min_epochs = 1000,
    # max_epochs = 600,
    log_every_n_steps = 1,

    # Instantiate Early Stopping Callback
    callbacks = [StartTrainingCallback(), StartTestingCallback(),
                 EarlyStopping(monitor='train_loss', mode='min', min_delta=0, patience=200, verbose=True)
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
  save_model(model_path, 'model.pth', model)
  # save_model('/home/fra/CodeFolderS/modelli', 'model.pth', model)
  # torch.save(model.state_dict(), '/home/fra/CodeFolder/model/model.pth')

  test_accuracy = trainer.callback_metrics.get('test_accuracy', None)
  if test_accuracy is not None:
    print(f'Test Accuracy: {test_accuracy.item()}')

  print('\n   TREMINE MAIN')



if __name__ == '__main__':

  main()
