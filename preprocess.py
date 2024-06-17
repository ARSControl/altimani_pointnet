import os
import torch
from dataset import ModelNet10Dataset

def preprocess_and_save_data(root_dir, num_points, augment=False):
    train_dataset = ModelNet10Dataset(root_dir, split='train', num_points=num_points, augment=augment)
    test_dataset = ModelNet10Dataset(root_dir, split='test', num_points=num_points, augment=False)
    
    os.makedirs('processed', exist_ok=True)
    torch.save(train_dataset, 'processed/train_dataset.pt')
    torch.save(test_dataset, 'processed/test_dataset.pt')
    return train_dataset, test_dataset
