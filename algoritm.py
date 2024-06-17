
import os
import torch
import trimesh
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, random_split
from dataset import MyDataset
from network import NeuralClassifier, num_points, num_classes
from utils import FOLDER, FOLDER_MODEL, PATH, DEVICE
import pickle

class MyAlgo:

  """ 3D Training Class """
  def __init__(self, batch_size, optimizer = 'Adam', learning_rate =0.0005, loss_function = 'cross_entropy'):
    # Get Database and Model Path
    self.model_path = os.path.join(FOLDER_MODEL, 'model')
    # Prepare Dataloaders
    dataset_shapes = self.prepareDataloaders(batch_size)
    # Create Model
    self.createModel(dataset_shapes, optimizer, learning_rate, loss_function)
    
  def prepareDataloaders(self, batch_size: int) -> Tuple[torch.Size, torch.Size]:
    """ Prepare Dataloaders """
    processed_data_path = 'processati/processed_data.pkl'

    # Check if processed data exists
    if os.path.exists(processed_data_path):
        # Load processed data
        processed_data = load_processed_data(processed_data_path)
        train_sequences, train_labels, test_sequences, test_labels = processed_data
    else:
        # Process data
        train_datacare = self.parse_dataset(num_points, mode='train')
        test_datacare = self.parse_dataset(num_points, mode='test')
  
        train_sequences, train_labels, _ = train_datacare
        test_sequences, test_labels, _ = test_datacare

        # Save processed data
        processed_data = (train_sequences, train_labels, test_sequences, test_labels)
        save_processed_data(processed_data, processed_data_path)

    train_dataset = MyDataset(train_sequences, train_labels)
    test_dataset = MyDataset(test_sequences, test_labels)

    self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataset.getInputShape(), train_dataset.getOutputShape()


  def getDataloaders(self) -> Tuple[DataLoader, DataLoader]:

    """ Get Dataloaders """
    
    return self.train_dataloader, self.test_dataloader
  
  def createModel(self, dataset_shape: Tuple[torch.Size, torch.Size], optimizer='Adam', lr=0.0005, loss_function='cross_entropy'):
    # Get Input and Output Sizes from Dataset Shapes
    input_size, output_size = torch.Size(list(dataset_shape[0])[1:]), torch.Size(list(dataset_shape[1])[1:])
  
    print('\nCreate Model: algoritm.py')   
    # DEBUG    \nInput Shape: {dataset_shape[0]} | Output Shape: {dataset_shape[1]}')
    # print(f'Input Size: {input_size} | Output Size: {output_size}')
  
    # Create NeuralNetwork Model
    self.model = NeuralClassifier(input_size, output_size, optimizer, lr, loss_function)
    self.model.to(DEVICE)
  
  def getModel(self) -> Tuple[NeuralClassifier, str]:
  
    """ Get NN Model and Model Path """
  
    return self.model
  
  
  def parse_dataset(self, num_points, mode='train') -> Tuple[np.ndarray, np.ndarray, dict]:

    # Parse the dataset by loading point clouds and labels
    # from the specified directories for training or testing
        
    print('\nthe PATH IS: ',os.listdir(PATH))
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
    #loads points and their relative label
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
  

def save_processed_data(data, filename):
  with open(filename, 'wb') as file:
    pickle.dump(data, file)


def load_processed_data(filename):
  with open(filename, 'rb') as file:
     data = pickle.load(file)
  return data


def load_model_from_file(model_folder, model_filename, batch_size):
  checkpoint = torch.load(os.path.join(model_folder, model_filename))
  # look for keys checkpoint
  if 'input_shape' in checkpoint and 'output_shape' in checkpoint and 'model_state_dict' in checkpoint:
    # take dimension of input and output
    input_shape = checkpoint['input_shape']
    output_shape = checkpoint['output_shape']
    # if are int,convert into tuple
    if not isinstance(input_shape, tuple):
      input_shape = (input_shape,)

    if not isinstance(output_shape, tuple):
      output_shape = (output_shape,)
    
    #create an istance of the model
    model = NeuralClassifier(input_shape, output_shape)
    #Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
      
    return model

  else:
    raise ValueError("chekpoint do not have the right keys.")
    