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
