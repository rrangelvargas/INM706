import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HybridCNNLSTM(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(HybridCNNLSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128)
        )
        
        self.feature_size = 128
        self.sequence_length = (150 // 8) * (150 // 8)
        
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        
    def forward(self, x):
        x = self.cnn(x)
        
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.sequence_length, self.feature_size)
        
        x, _ = self.rnn(x)

        x = x[:, -1, :]

        x = self.classifier(x)
        
        return x
    
    def run(self, train_loader, test_loader, criterion, optimizer, epochs=30, save_model=True, logger=None, model_idx=0):
        best_acc = 0
        train_losses = []
        train_accs = []
        test_accs = []

        progress_bar = tqdm(range(epochs), desc='Training Progress')
        for epoch in progress_bar:
            train_loss, train_acc = train_epoch(self, train_loader, criterion, optimizer)
            test_acc = evaluate(self, test_loader)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            progress_bar.set_postfix({
                'Epoch': f'{epoch+1}/{epochs}',
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.2f}%',
                'Test Acc': f'{test_acc:.2f}%'
            })

            if save_model: 
                if test_acc > best_acc:
                    best_acc = test_acc
                torch.save(self.state_dict(), f'output/saved_models/best_lstm_model_{model_idx}.pth')

            if logger:
                logger.log({
                    "LSTM/Train Loss": train_loss,
                    "LSTM/Train Accuracy": train_acc,
                    "LSTM/Test Accuracy": test_acc,
                    "epoch": epoch
                })

                logger.log({
                    "LSTM/Loss vs Epochs": wandb.plot.line_series(
                        xs=list(range(len(train_losses))),
                        ys=[train_losses],
                        keys=["Training Loss"],
                        title="Loss vs Epochs",
                        xname="Epoch"
                    ),
                    "LSTM/Accuracy vs Epochs": wandb.plot.line_series(
                        xs=list(range(len(train_accs))),
                        ys=[train_accs, test_accs],
                        keys=["Training Accuracy", "Testing Accuracy"],
                        title="Accuracy vs Epochs",
                        xname="Epoch"
                    )
                })

        return train_loss, train_acc, test_acc


class HybridCNNGRU(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(HybridCNNGRU, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128)
        )
        
        self.feature_size = 128
        self.sequence_length = (150 // 8) * (150 // 8)
        
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        
    def forward(self, x):
        x = self.cnn(x)
        
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.sequence_length, self.feature_size)
        
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        
        x = self.classifier(x)
        
        return x
    
    def run(self, train_loader, test_loader, criterion, optimizer, epochs=30, save_model=True, logger=None, model_idx=0):
        best_acc = 0
        train_losses = []
        train_accs = []
        test_accs = []

        progress_bar = tqdm(range(epochs), desc='Training Progress')
        for epoch in progress_bar:
            train_loss, train_acc = train_epoch(self, train_loader, criterion, optimizer)
            test_acc = evaluate(self, test_loader)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            progress_bar.set_postfix({
                'Epoch': f'{epoch+1}/{epochs}',
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.2f}%',
                'Test Acc': f'{test_acc:.2f}%'
            })

            if save_model: 
                if test_acc > best_acc:
                    best_acc = test_acc
                torch.save(self.state_dict(), f'output/saved_models/best_gru_model_{model_idx}.pth')

            if logger:
                logger.log({
                    "GRU/Train Loss": train_loss,
                    "GRU/Train Accuracy": train_acc,
                    "GRU/Test Accuracy": test_acc,
                    "epoch": epoch
                })

                logger.log({
                    "GRU/Loss vs Epochs": wandb.plot.line_series(
                        xs=list(range(len(train_losses))),
                        ys=[train_losses],
                        keys=["Training Loss"],
                        title="Loss vs Epochs",
                        xname="Epoch"
                    ),
                    "GRU/Accuracy vs Epochs": wandb.plot.line_series(
                        xs=list(range(len(train_accs))),
                        ys=[train_accs, test_accs],
                        keys=["Training Accuracy", "Testing Accuracy"],
                        title="Accuracy vs Epochs",
                        xname="Epoch"
                    )
                })

        return train_loss, train_acc, test_acc


class HybridCNNGRUWithAttention(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3, l2_lambda=0.01, use_l2=True):
        super(HybridCNNGRUWithAttention, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128)
        )
        
        self.feature_size = 128
        self.sequence_length = (150 // 8) * (150 // 8)
        
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
        self.attention = Attention(hidden_size=128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        self.l2_lambda = l2_lambda
        self.use_l2 = use_l2

    def forward(self, x):
        x = self.cnn(x)
        
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.sequence_length, self.feature_size)
        
        x, _ = self.rnn(x)

        x = self.attention(x)

        x = self.classifier(x)
        
        if self.use_l2:
            l2_reg = torch.tensor(0., requires_grad=True).to(x.device)
            for param in self.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)
            
            x = x + self.l2_lambda * l2_reg
        
        return x
    
    def run(self, train_loader, test_loader, criterion, optimizer, epochs=30, save_model=True, logger=None, model_idx=0):
        best_acc = 0
        train_losses = []
        train_accs = []
        test_accs = []

        progress_bar = tqdm(range(epochs), desc='Training Progress')
        for epoch in progress_bar:
            train_loss, train_acc = train_epoch(self, train_loader, criterion, optimizer)
            test_acc = evaluate(self, test_loader)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            progress_bar.set_postfix({
                'Epoch': f'{epoch+1}/{epochs}',
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.2f}%',
                'Test Acc': f'{test_acc:.2f}%'
            })

            if save_model:
                if test_acc > best_acc:
                    best_acc = test_acc
                torch.save(self.state_dict(), f'output/saved_models/best_gru_with_attention_model_{model_idx}.pth')

            if logger:
                logger.log({
                    "GRU+Attention/Train Loss": train_loss,
                    "GRU+Attention/Train Accuracy": train_acc,
                    "GRU+Attention/Test Accuracy": test_acc,
                    "epoch": epoch
                })

                logger.log({
                    "GRU+Attention/Loss vs Epochs": wandb.plot.line_series(
                        xs=list(range(len(train_losses))),
                        ys=[train_losses],
                        keys=["Training Loss"],
                        title="Loss vs Epochs",
                        xname="Epoch"
                    ),
                    "GRU+Attention/Accuracy vs Epochs": wandb.plot.line_series(
                        xs=list(range(len(train_accs))),
                        ys=[train_accs, test_accs],
                        keys=["Training Accuracy", "Testing Accuracy"],
                        title="Accuracy vs Epochs",
                        xname="Epoch"
                    )
                })

        return train_loss, train_acc, test_acc

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.fc(x).squeeze(2), dim=1)
        context = torch.sum(x * attn_weights.unsqueeze(2), dim=1)
        return context


# Training function
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total


# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total