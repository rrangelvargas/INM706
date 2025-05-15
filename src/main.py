from train import train
import torch
from dataset import get_train_test_split
import pandas as pd
import os
import wandb
import numpy as np

def get_device_info():
    device_info = {
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_device_name": None
    }
    
    if torch.cuda.is_available():
        try:
            device_info.update({
                "device": "cuda",
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_device_memory": torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
            })
        except Exception as e:
            print(f"Warning: CUDA initialization error: {str(e)}")
            print("Falling back to CPU")
    
    return device_info
    with wandb.init(
        project="inm706",
        name="grid-search",
        config={
            "description": "Grid search plots"
        }
    ) as comparison_run:        
        # Create a bar chart of final test accuracies for top 5
        final_accuracies = [m['final_test_acc'] for m in top_5_metrics]
        configs = [f"{m['config']}\n(lr={m['learning_rate']}, dr={m['dropout_rate']}, wd={m['weight_decay']}, bs={m['batch_size']})" 
                  for m in top_5_metrics]
        comparison_run.log({
            "Grid Search/Final Test Accuracies": wandb.plot.bar(
                wandb.Table(
                    data=[[config, acc] for config, acc in zip(configs, final_accuracies)],
                    columns=["Configuration", "Final Test Accuracy"]
                ),
                "Configuration",
                "Final Test Accuracy",
                title="Final Test Accuracies - Top 5 Configurations"
            )
        })
        
        # Create a bar chart of best accuracies for top 5
        best_accuracies = [m['best_acc'] for m in top_5_metrics]
        comparison_run.log({
            "Grid Search/Best Test Accuracies": wandb.plot.bar(
                wandb.Table(
                    data=[[config, acc] for config, acc in zip(configs, best_accuracies)],
                    columns=["Configuration", "Best Test Accuracy"]
                ),
                "Configuration",
                "Best Test Accuracy",
                title="Best Test Accuracies - Top 5 Configurations"
            )
        })

if __name__ == "__main__":
    # Get device information
    device_info = get_device_info()
    print(f"Using {device_info['device'].upper()}")
    
    # Hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1] 
    dropout_rates = [0.3, 0.5, 0.7]
    weight_decays = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]

    X_train, X_test, y_train, y_test = get_train_test_split()
    
    results = []
    all_metrics = []  # Store metrics for all configurations

    try:
        idx = 0

        for learning_rate in learning_rates:
            for dropout_rate in dropout_rates:
                for weight_decay in weight_decays:
                    for batch_size in batch_sizes:
                        
                        # Clear GPU memory if available
                        if device_info['cuda_available']:
                            try:
                                torch.cuda.empty_cache()
                            except Exception as e:
                                print(f"Warning: Could not clear GPU memory: {str(e)}")
                        
                        print("----------------------------------------------------------------")
                        print("Configuration:")
                        print(f"Learning Rate: {learning_rate}")
                        print(f"Dropout Rate: {dropout_rate}")
                        print(f"Weight Decay: {weight_decay}")
                        print(f"Batch Size: {batch_size}")
                        print("--------------------------------")
                        
                        # Create a unique experiment name for this configuration
                        experiment_name = f"lr{learning_rate}_dr{dropout_rate}_wd{weight_decay}_bs{batch_size}"
                        
                        # Initialize wandb with the configuration using context manager
                        with wandb.init(
                            project="inm706",
                            name=f"music-genre-classification-{experiment_name}",
                            config={
                                "learning_rate": learning_rate,
                                "dropout_rate": dropout_rate,
                                "weight_decay": weight_decay,
                                "batch_size": batch_size,
                                "model_idx": idx,
                                "device_info": device_info
                            }
                        ) as run:
                            metrics = train(
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
                                model_idx=idx,
                                logger=run
                            )
                            
                            # Store metrics for comparison
                            all_metrics.append({
                                'config': experiment_name,
                                'test_accs': metrics['test_accs'],
                                'train_losses': metrics['train_losses'],
                                'final_test_acc': metrics['final_test_acc'],
                                'best_acc': metrics['best_acc'],
                                'learning_rate': learning_rate,
                                'dropout_rate': dropout_rate,
                                'weight_decay': weight_decay,
                                'batch_size': batch_size
                            })
                            
                            results.append({
                                'model_idx': idx,
                                'learning_rate': learning_rate,
                                'dropout_rate': dropout_rate,
                                'weight_decay': weight_decay,
                                'batch_size': batch_size,
                                'train_loss': metrics['final_train_loss'],
                                'train_accuracy': metrics['final_train_acc'],
                                'test_accuracy': metrics['final_test_acc'],
                                'best_accuracy': metrics['best_acc']
                            })
                        
                        idx += 1
                        print("----------------------------------------------------------------")
        
        # Sort configurations by best accuracy and get top 5
        all_metrics.sort(key=lambda x: x['best_acc'], reverse=True)
        top_5_metrics = all_metrics[:5]
        
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('output/training_results.csv', index=False)