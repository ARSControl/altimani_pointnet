import torch
from torch.utils.data import DataLoader
from algoritm import MyAlgo, load_model_from_file
from model import PointNetClassHead
from dataset import collate_fn
import torch.optim as optim
import torch.nn.functional as F
import logging
import os

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for i, (points, labels) in enumerate(train_loader):
        points = points.permute(0, 2, 1)
        optimizer.zero_grad()
        outputs, _, _ = model(points)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (i + 1) % 10 == 0:  # Stampa il log ogni 10 batch
            logging.info(f'Epoch [{epoch + 1}/100], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    logging.info(f'Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}')

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in test_loader:
            points = points.permute(0, 2, 1)
            outputs, _, _ = model(points)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    logging.info(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

def main():
    num_epochs = 20
    num_points = 512  # Specifica il numero di punti desiderato
    algo = MyAlgo(PointNetClassHead(k=10), 'ModelNet10', num_points=num_points)
    
    # Carica il modello pre-allenato se esiste
    model = PointNetClassHead(k=10)
    if os.path.exists('best_model.pth'):
        logging.info('Loading pre-trained model...')
        model.load_state_dict(torch.load('best_model.pth'))
    else:
        logging.info('No pre-trained model found. Training from scratch...')
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Caricamento dei dati pre-elaborati
    if os.path.exists('processed/train_dataset.pt') and os.path.exists('processed/test_dataset.pt'):
        logging.info('Loading pre-processed datasets...')
        train_dataset = torch.load('processed/train_dataset.pt')
        test_dataset = torch.load('processed/test_dataset.pt')
    else:
        logging.info('No pre-processed data found. Processing data...')
        train_dataset = algo.load_data(split='train')
        test_dataset = algo.load_data(split='test')
        os.makedirs('processed', exist_ok=True)
        torch.save(train_dataset, 'processed/train_dataset.pt')
        torch.save(test_dataset, 'processed/test_dataset.pt')
    
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, epoch)
        accuracy = test(model, test_loader)
        
        # Salva il modello se l'accuratezza migliora
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f'New best model saved with accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
