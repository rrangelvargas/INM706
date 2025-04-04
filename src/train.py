import torch
import torch.nn as nn
import torch.optim as optim
from model import HybridCNNGRUWithAttention
from dataset import load_dataset

def train(
        learning_rate,
        dropout_rate,
        weight_decay,
        batch_size,
        X_train,
        X_test,
        y_train,
        y_test,
        l2_lambda,
        use_l2,
        model_idx
    ):

    train_loader, test_loader = load_dataset(X_train, X_test, y_train, y_test, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model = HybridCNNGRUWithAttention(
        num_classes=10,
        dropout_rate=dropout_rate,
        l2_lambda=l2_lambda,
        use_l2=use_l2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loss, train_acc, test_acc = model.run(
        train_loader,
        test_loader,
        criterion,
        optimizer,
        epochs=30,
        save_model=True,
        model_idx=model_idx,
        # logger=logger
    )

    return train_loss, train_acc, test_acc