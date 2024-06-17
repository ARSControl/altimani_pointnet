
import torch
from dataset import ModelNet10Dataset

class MyAlgo:

    """ Class to encapsulate the model and dataset loading """

    def __init__(self, model, dataset_path, num_points=1024):

        # Model, Dataset and number of points
        self.model, self.dataset_path, self.num_points = model, dataset_path, num_points

    def load_data(self, split='train'):

        # Load the Dataset
        return ModelNet10Dataset(self.dataset_path, split, self.num_points)

def load_model_from_file(filepath):

    # Load the model from the file
    model = torch.load(filepath)

    # Set the model to evaluation mode
    model.eval()

    return model