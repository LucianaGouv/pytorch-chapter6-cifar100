#!/usr/bin/env python3
"""Run a short demo of the Chapter6 experiments to stream logs in terminal.
This script imports utilities from Chapter6_CIFAR100_Complete.py and runs
short Adam training (3 epochs) so you can follow progress in real time.
"""
import time
import torch
import importlib.util
import os

# Import Chapter6_CIFAR100_Complete.py dynamically by path to avoid module import issues
spec_path = os.path.join(os.path.dirname(__file__), '..', 'Chapter6_CIFAR100_Complete.py')
spec_path = os.path.abspath(spec_path)
spec = importlib.util.spec_from_file_location('ch6mod', spec_path)
ch6mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch6mod)

# pull the required functions/classes from the loaded module
prepare_cifar100_data = ch6mod.prepare_cifar100_data
LeNet5_CIFAR100 = ch6mod.LeNet5_CIFAR100
train_model = ch6mod.train_model
plot_training_history = ch6mod.plot_training_history
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print('Demo: treino curto (Adam) — 3 épocas')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # prepare data with smaller batch and fewer workers for demo
    train_loader, val_loader, test_loader = prepare_cifar100_data(batch_size=128, val_split=0.1, data_augmentation=False)

    model = LeNet5_CIFAR100(num_classes=100, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train for few epochs and print progress
    history = train_model(model, train_loader, val_loader, criterion, optimizer,
                          num_epochs=3, device=device, verbose=True)

    # save a small training plot
    fig = plot_training_history(history, title='Demo - Adam (3 epochs)')
    fig.savefig('demo_training_adam_3ep.png')
    print('Saved demo figure: demo_training_adam_3ep.png')


if __name__ == '__main__':
    main()
