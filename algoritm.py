# import os, torch, trimesh
# import numpy as np
# from typing import Tuple

# from pytorch_lightning import LightningModule
# from torch.utils.data import DataLoader, random_split

# from dataset import MyDataset
# from network import NeuralClassifier, num_points
# from utils import FOLDER, PATH, DEVICE
# from numba import njit

# class MyAlgo:

#   """ 3D Training Class """

#   def __init__(self, batch_size, optimizer, learning_rate, loss_function):

#     # Get Database and Model Path
#     self.model_path = os.path.join(FOLDER, 'model')

#     # Prepare Dataloaders
#     dataset_shapes = self.prepareDataloaders(batch_size)

#     # Create Model
#     self.createModel(dataset_shapes, optimizer, learning_rate, loss_function)

#   def prepareDataloaders(self, batch_size: int) -> Tuple[torch.Size, torch.Size]:
#     """ Prepare Dataloaders """
#     datacare = self.parse_dataset(num_points)
#     # sequences, labels = self.parse_dataset(num_points) 
#     print("dentro dataloaders")

#     sequences, labels, _ = datacare
    
#     print('sembra OK')

#     dataset= MyDataset(sequences, labels)
    
#     # assert sequences.shape[0] == labels.shape[0], 'Sequences and Labels must have the same length'
#     # assert sequences.shape[1:] == torch.Size(DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count())), 'Dataset Input Shape must be equal to Sequences Shape'
#     # assert labels.shape[1:] == torch.Size(DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count())), 'Dataset Output Shape must be equal to Labels Shape'    
#     assert sequences.shape[0] == labels.shape[0], 'Sequences and Labels must have the same length'
    
#     assert torch.Size(sequences.reshape(sequences.shape[0], -1).shape) == dataset.getInputShape(), 'Dataset Input Shape must be equal to Sequences Shape'
#     assert labels.shape[0] == dataset.getOutputShape()[0], 'Dataset Output Shape must be equal to Labels Shape'

#     # # # Split Dataset
#     # # assert train_set_size + test_set_size <= 1, 'Train + Test Set Size must be less than 1'
#     # # train_data, test_data = random_split(dataset, [train_set_size, test_set_size], generator=torch.Generator())
#     # # assert len(train_data) + len(test_data) == len(dataset), 'Train + Validation + Test Set Size must be equal to Dataset Size'

#     # # Create data loaders for training and testing
#     # self.train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)
#     # self.test_dataloader  = DataLoader(test_data,  batch_size=batch_size, num_workers=os.cpu_count(), shuffle=False)

#     self.train_dataloader = DataLoader(dataset, batch_size=batch_size)
#     self.test_dataloader  = DataLoader(dataset, batch_size=batch_size)
#     # self.train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)
#     # self.test_dataloader  = DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=False)
       
#     return dataset.getInputShape(), dataset.getOutputShape()

#   def getDataloaders(self) -> Tuple[DataLoader, DataLoader]:
#     """ Get Dataloaders """
#     return self.train_dataloader, self.test_dataloader

#   def createModel(self, dataset_shape: Tuple[torch.Size, torch.Size], optimizer='SDG', lr=0.0005, loss_function='cross_entropy'):
#     # Get Input and Output Sizes from Dataset Shapes
#     input_size, output_size = torch.Size(list(dataset_shape[0])[1:]), torch.Size(list(dataset_shape[1])[1:])
#     # input_size, output_size = dataset_shape

#     print(f'\nCreate Model:\nInput Shape: {dataset_shape[0]} | Output Shape: {dataset_shape[1]}')
#     print(f'Input Size: {input_size} | Output Size: {output_size}')

#     # Create NeuralNetwork Model
#     self.model = NeuralClassifier(input_size, output_size, optimizer, lr, loss_function)
#     self.model.to(DEVICE)

#   def getModel(self) -> Tuple[LightningModule, str]:

#     """ Get NN Model and Model Path """

#     return self.model        #, self.model_path

#   def parse_dataset(self, num_points) -> Tuple[np.ndarray, np.ndarray, dict]:

#     print(os.listdir(PATH))
#     folders = [dir for dir in sorted(os.listdir(PATH)) if os.path.isdir(os.path.join(PATH, dir))]

#     class_map = {}

#     points, labels = [], []

#     for i, folder in enumerate(folders):
#         print(f"Processing class: {os.path.basename(folder)}")

        
#         class_map[i] = folder.split("/")[-1]

#         train_files = os.path.join(f'{PATH}/{folder}', "train/")
#         test_files = os.path.join(f'{PATH}/{folder}', "test/")
        
#         points_train, labels_train = self.load_files(i, train_files, num_points)
#         points_test, labels_test = self.load_files(i, test_files, num_points)

#         points.extend(points_train)
#         labels.extend(labels_train)
    

#     return np.array(points), np.array(labels), class_map

#   def load_files(self, i, T_files, num_points):
#     # points, labels = np.asarray([]), np.asarray([])
#     points, labels = [],[]

#     for f in os.listdir(T_files):
        
#         points.append(trimesh.load(f"{T_files}/{f}").sample(num_points))
#         labels.append(i)

#     return points, labels 

import os
import torch
import trimesh
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, random_split
from dataset import MyDataset
from network import NeuralClassifier, num_points
from utils import FOLDER, PATH, DEVICE

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

    train_datacare = self.parse_dataset(num_points, mode='train')
    test_datacare = self.parse_dataset(num_points, mode='test')
  
    train_sequences, train_labels, _ = train_datacare
    test_sequences, test_labels, _ = test_datacare

    train_dataset = MyDataset(train_sequences, train_labels)
    test_dataset = MyDataset(test_sequences, test_labels)

    self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataset.getInputShape(), train_dataset.getOutputShape()
  
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
  
  def getModel(self) -> Tuple[NeuralClassifier, str]:
  
    """ Get NN Model and Model Path """
  
    return self.model
  
  def parse_dataset(self, num_points, mode='train') -> Tuple[np.ndarray, np.ndarray, dict]:
  
    print(os.listdir(PATH))
    folders = [dir for dir in sorted(os.listdir(PATH)) if os.path.isdir(os.path.join(PATH, dir))]
  
    class_map = {}
    points, labels = [], []
  
    for i, folder in enumerate(folders):
      print(f"Processing class: {os.path.basename(folder)}")
      class_map[i] = folder.split("/")[-1]
      data_folder = os.path.join(PATH, folder, mode)
      points_data, labels_data = self.load_files(i, data_folder, num_points)
      points.extend(points_data)
      labels.extend(labels_data)

    return np.array(points), np.array(labels), class_map
  
  def load_files(self, i, T_files, num_points):
    points, labels = [], []
    
    for f in os.listdir(T_files):
      file_path = os.path.join(T_files, f)
      
      try:
          
        # Load 3D point cloud using trimesh
        point_cloud = trimesh.load(file_path).sample(num_points)
        points.append(point_cloud)
        labels.append(i)

      except Exception as e:
        print(f"ERROR loading file {file_path}: {e}")

    return points, labels
