# Demo curto: comparar SGD (momentum) vs SGD (Nesterov) — salva figura 'sgd_nesterov_demo.png'
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reusar definição LeNetLike se já existir, caso contrário definir rapidamente
try:
    LeNetLike
except NameError:
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

# Data loader: usar uma pequena quantidade para demo rápido
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()

def short_run_sgd_variants(nesterov=False, lr=1e-2, max_batches=40, epochs=3):
    torch.manual_seed(0)
    model = LeNetLike(num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=nesterov)
    losses = []
    it = 0
    start = time.time()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            it += 1
            if it >= max_batches:
                break
        if it >= max_batches:
            break
    return losses, time.time() - start

# Rodar comparações: SGD+momentum (sem nesterov) vs SGD+Nesterov
sgd_losses, sgd_time = short_run_sgd_variants(nesterov=False, lr=1e-2, max_batches=40, epochs=3)
nesterov_losses, nesterov_time = short_run_sgd_variants(nesterov=True, lr=1e-2, max_batches=40, epochs=3)

plt.figure(figsize=(8,4))
plt.plot(sgd_losses, label=f'SGD+momentum (lr=1e-2)')
plt.plot(nesterov_losses, label=f'SGD+Nesterov (lr=1e-2)')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('SGD Momentum vs SGD Nesterov — Demo curto')
plt.legend()
plt.tight_layout()
plt.savefig('sgd_nesterov_demo.png', dpi=150)
plt.close()
print(f'SGD: {len(sgd_losses)} iters, time={sgd_time:.1f}s, final loss={sgd_losses[-1]:.4f}')
print(f'Nesterov: {len(nesterov_losses)} iters, time={nesterov_time:.1f}s, final loss={nesterov_losses[-1]:.4f}')
