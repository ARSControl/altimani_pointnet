import torch
import os
import trimesh
import numpy as np
from torch.utils.data import Dataset

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / furthest_distance
    return points

def jitter_points(points, sigma=0.01, clip=0.05):
    N, C = points.shape
    jittered_data = np.clip(sigma * np.random.randn(N, C), -clip, clip)
    jittered_data += points
    return jittered_data

# def collate_fn(batch):
#     points, labels = zip(*batch)
#     points = np.array(points)
#     labels = np.array(labels)
#     return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
def collate_fn(batch):
    points, labels = zip(*batch)
    points = [torch.tensor(point, dtype=torch.float32) for point in points]
    labels = torch.tensor(labels, dtype=torch.long)
    return torch.stack(points), labels


class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=512, augment=False):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.classes = sorted(os.listdir(root_dir))
        self.files = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls, split)
            self.files += [(os.path.join(cls_dir, f), cls) for f in os.listdir(cls_dir) if f.endswith('.off')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, cls = self.files[idx]
        point_cloud = self.load_point_cloud(file_path)
        if self.augment:
            point_cloud = self.augment_point_cloud(point_cloud)
        label = self.classes.index(cls)
        return torch.tensor(point_cloud, dtype=torch.float32).transpose(0, 1), label

    def load_point_cloud(self, file_path):
        mesh = trimesh.load_mesh(file_path)
        points = mesh.sample(self.num_points)
        # Ensure points is a numpy array with dtype float32
        points = np.array(points, dtype=np.float32)
        return points

    def augment_point_cloud(self, point_cloud):
        # for add trasformation, just in case
        return point_cloud
