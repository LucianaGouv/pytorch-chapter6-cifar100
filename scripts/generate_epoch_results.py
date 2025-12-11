#!/usr/bin/env python3
"""Run a short deterministic training on CIFAR-100 to generate epoch metrics CSV.

This script is conservative (few batches) so it runs quickly for notebook/article tables.
It writes `results/epoch_history.csv`.
"""

import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class LeNetLike(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)

    # Small quick loaders: limit batches per epoch via sampler slicing
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    epochs = 3
    max_batches_per_epoch = 80  # keep short

    model = LeNetLike(num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_count += 1
            if batch_count >= max_batches_per_epoch:
                break

        train_loss = running_loss / batch_count
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Validation (quick: limit to a few batches)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                val_batches += 1
                if val_batches >= 20:
                    break

        val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0

        # Learning rate (support for param groups)
        current_lr = optimizer.param_groups[0]['lr']

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, lr={current_lr}")

    # Save CSV
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', 'epoch_history.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i],
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.3f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['val_acc'][i]:.3f}",
                f"{history['lr'][i]:.6g}",
            ])

    print(f"Wrote {csv_path}")


if __name__ == '__main__':
    main()
