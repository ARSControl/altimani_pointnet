
import torch,os
import trimesh
import numpy as np
from torch.utils.data import Dataset

class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=512):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.classes = sorted(os.listdir(root_dir))
        self.files = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name, split)
            self.files += [(os.path.join(class_dir, f), class_name) for f in os.listdir(class_dir) if f.endswith('.off')]
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, class_name = self.files[idx]
        mesh = trimesh.load(file_path)
        points = mesh.sample(self.num_points)  # Campionare il numero di punti specificato
        points = np.array(points, dtype=np.float32)
        
        class_id = self.classes.index(class_name)
        return points, class_id

def collate_fn(batch):
    points, labels = zip(*batch)
    points = np.array(points)
    labels = np.array(labels)
    return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
