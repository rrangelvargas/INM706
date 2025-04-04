from train import train
import torch
from dataset import get_train_test_split
import pandas as pd
from logger import Logger
import os

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("Using CPU")

    # wandb_logger = Logger(f"inm706_music_genre_classification", project='inm706_project')
    # logger = wandb_logger.get_logger()

    learning_rates = [0.001, 0.01, 0.1] 
    dropout_rates = [0.3, 0.5, 0.7]
    weight_decays = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]

    X_train, X_test, y_train, y_test = get_train_test_split()
    
    results = []

    idx = 0

    for learning_rate in learning_rates:
        for dropout_rate in dropout_rates:
            for weight_decay in weight_decays:
                for batch_size in batch_sizes:
                    
                    torch.cuda.empty_cache()

                    # logger.log(f"Learning Rate: {learning_rate}")
                    # logger.log(f"Dropout Rate: {dropout_rate}")
                    # logger.log(f"Weight Decay: {weight_decay}")
                    # logger.log(f"Batch Size: {batch_size}")
                    
                    print("----------------------------------------------------------------")
                    print("Configuration:")
                    print(f"Learning Rate: {learning_rate}")
                    print(f"Dropout Rate: {dropout_rate}")
                    print(f"Weight Decay: {weight_decay}")
                    print(f"Batch Size: {batch_size}")
                    print("--------------------------------")
                    
                    train_loss, train_acc, test_acc = train(
                        learning_rate,
                        dropout_rate,
                        weight_decay,
                        batch_size,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        use_l2=True,
                        l2_lambda=0.01,
                        model_idx=idx
                    )
                    
                    results.append({
                        'model_idx': idx,
                        'learning_rate': learning_rate,
                        'dropout_rate': dropout_rate,
                        'weight_decay': weight_decay,
                        'batch_size': batch_size,
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc
                    })

                    idx += 1
                    
                    print("----------------------------------------------------------------")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('output/training_results.csv', index=False)