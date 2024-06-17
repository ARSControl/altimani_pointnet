
import torch, os
from torch.utils.data import DataLoader
from algoritm import MyAlgo, load_model_from_file
from model import PointNet
from dataset import collate_fn
import torch.optim as optim
import torch.nn.functional as F
import logging

# Get the current file path
main_path = os.path.dirname(os.path.abspath(__file__))

# Get Torch Device ('cuda' or 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


min_epochs, max_epochs = 10000, 15000   ##mod
min_delta, patience = 0, 1000
fast_dev_run = True


MODEL_PATH = os.path.join(main_path, 'best_model.pth')
PROCESSED_DATA_DIR = os.path.join(main_path, 'processed')

def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0

    for i, (points, labels) in enumerate(train_loader):
        points = points.permute(0, 2, 1).to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(points)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (i + 1) % 10 == 0:  # Stampa il log ogni 10 batch
            logging.info(f'Epoch [{epoch + 1}/{100}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    logging.info(f'Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}')

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in test_loader:
            points = points.permute(0, 2, 1)
            outputs = model(points)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    logging.info(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

def load_or_process_data(algo, split='train'):
    processed_data_path = os.path.join(PROCESSED_DATA_DIR, f'{split}_data.pt')
    
    if os.path.exists(processed_data_path):
        logging.info(f'Loading processed {split} data from {processed_data_path}')
        data = torch.load(processed_data_path)
    else:
        logging.info(f'Processing {split} data...')
        data = algo.load_data(split=split)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        torch.save(data, processed_data_path)
        logging.info(f'Processed {split} data saved to {processed_data_path}')
    
    return data



def main():
    num_epochs = 100
    num_points = 512  # Specifica il numero di punti desiderato
    root_dir = os.path.join(main_path, 'ModelNet10')
    algo = MyAlgo(PointNet(k=10), root_dir, num_points=num_points)
    train_dataset = algo.load_data(split='train')
    test_dataset = algo.load_data(split='test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    if os.path.exists(MODEL_PATH):
        logging.info(f'Loading pre-trained model from {MODEL_PATH}')
        model = PointNet(k=10).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model = PointNet(k=10).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, epoch)
        accuracy = test(model, test_loader)
        
        # Salva il modello se l'accuratezza migliora
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'best_model.pth'))
            logging.info(f'New best model saved with accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
