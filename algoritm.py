import torch
from dataset import ModelNet10Dataset
from model import PointNetClassHead

class MyAlgo:
    def __init__(self, model, dataset_name, num_points=512):
        self.model = model
        self.dataset_name = dataset_name
        self.num_points = num_points

    def load_data(self, split):
        if self.dataset_name == 'ModelNet10':
            root_dir = '/home/fra/3Dpc_TEST/ModelNet10'
            root_dir = r'/home/davide/ROS/Altri Workspace/ARS-Control-Projects/Tesisti/Francesco Altimani/ROS Nodes/altimani_ws/3Dpc - v4 [12-06]/V_tnet/ModelNet10'
            return ModelNet10Dataset(root_dir, split=split, num_points=self.num_points, augment=True)
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset_name}')

def load_model_from_file(file_path):
    model = PointNetClassHead(k=10)
    model.load_state_dict(torch.load(file_path))
    return model
