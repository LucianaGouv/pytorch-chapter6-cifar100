"""
Capítulo 6 - Deep Learning com PyTorch aplicado ao CIFAR-100
Trabalho Final - Projeto de Sistemas Baseados em Aprendizado de Máquinas
Universidade Federal do Rio Grande do Norte (UFRN)
Professor: Ivanovich
Aluna: Luciana Gouveia
Data: Dezembro de 2025

Este arquivo contém todo o código necessário para reproduzir os experimentos
do Capítulo 6 aplicados ao dataset CIFAR-100, incluindo:
- EWMA (Exponentially Weighted Moving Averages)
- Otimizadores (Adam, SGD, SGD+Momentum, SGD+Nesterov)
- Visualização de gradientes
- Learning Rate Schedulers
- Captura de ativações com hooks
"""

# ================================================================================
# IMPORTS
# ================================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ReduceLROnPlateau, 
    CyclicLR, LambdaLR, CosineAnnealingLR
)

# Configurar device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ================================================================================
# 1. EWMA - EXPONENTIALLY WEIGHTED MOVING AVERAGES
# ================================================================================

def calc_ewma(values, beta=0.9):
    """
    Calcula EWMA sem bias correction
    
    Args:
        values: lista ou array de valores
        beta: fator de decaimento (default: 0.9)
    
    Returns:
        np.array com valores do EWMA
    """
    ewma = []
    v = 0
    
    for value in values:
        v = beta * v + (1 - beta) * value
        ewma.append(v)
    
    return np.array(ewma)


def calc_corrected_ewma(values, beta=0.9):
    """
    Calcula EWMA com bias correction
    
    Args:
        values: lista ou array de valores
        beta: fator de decaimento (default: 0.9)
    
    Returns:
        np.array com valores do EWMA corrigido
    """
    ewma = []
    v = 0
    
    for step, value in enumerate(values, 1):
        v = beta * v + (1 - beta) * value
        # Bias correction
        v_corrected = v / (1 - beta ** step)
        ewma.append(v_corrected)
    
    return np.array(ewma)


def calc_sma(values, window=19):
    """
    Calcula Simple Moving Average
    
    Args:
        values: lista ou array de valores
        window: tamanho da janela
    
    Returns:
        np.array com valores do SMA
    """
    sma = []
    for i in range(len(values)):
        if i < window:
            sma.append(np.mean(values[:i+1]))
        else:
            sma.append(np.mean(values[i-window+1:i+1]))
    
    return np.array(sma)


# ================================================================================
# 2. ARQUITETURA DO MODELO - LeNet-5 Adaptada para CIFAR-100
# ================================================================================

class LeNet5_CIFAR100(nn.Module):
    """
    LeNet-5 adaptada para CIFAR-100
    - Input: 3x32x32 (RGB)
    - Output: 100 classes
    """
    def __init__(self, num_classes=100, dropout_p=0.3):
        super(LeNet5_CIFAR100, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Camadas fully connected
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)
    
    def forward(self, x):
        # Conv1 + ReLU + MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Conv2 + ReLU + MaxPool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers com dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# ================================================================================
# 3. CAPTURA DE GRADIENTES E ATIVAÇÕES
# ================================================================================

class GradientCapture:
    """
    Classe para capturar gradientes durante o treinamento
    """
    def __init__(self, model):
        self.model = model
        self.gradients = {}
        self.hooks = []
    
    def register_hooks(self, layer_names):
        """
        Registra hooks para capturar gradientes de camadas específicas
        
        Args:
            layer_names: lista de nomes de camadas para monitorar
        """
        for name, param in self.model.named_parameters():
            if any(ln in name for ln in layer_names):
                def make_hook(n):
                    def hook(grad):
                        if n not in self.gradients:
                            self.gradients[n] = []
                        self.gradients[n].append(grad.cpu().clone().numpy())
                        return None  # Não modificar gradientes
                    return hook
                
                hook_fn = param.register_hook(make_hook(name))
                self.hooks.append(hook_fn)
    
    def remove_hooks(self):
        """Remove todos os hooks registrados"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class ActivationCapture:
    """
    Classe para capturar ativações intermediárias
    """
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, layer_names):
        """
        Registra hooks para capturar ativações de camadas específicas
        
        Args:
            layer_names: lista de nomes de módulos para monitorar
        """
        for name, module in self.model.named_modules():
            if name in layer_names:
                def make_hook(n):
                    def hook(module, input, output):
                        self.activations[n] = output.detach()
                    return hook
                
                hook_fn = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook_fn)
    
    def remove_hooks(self):
        """Remove todos os hooks registrados"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ================================================================================
# 4. PREPARAÇÃO DOS DADOS - CIFAR-100
# ================================================================================

def prepare_cifar100_data(batch_size=128, val_split=0.1, data_augmentation=True):
    """
    Prepara os datasets e dataloaders para CIFAR-100
    
    Args:
        batch_size: tamanho do batch
        val_split: proporção para validação
        data_augmentation: aplicar data augmentation no treino
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Normalização CIFAR-100
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    
    # Transformações de treino
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    # Transformações de teste
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Carregar datasets
    train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Split treino/validação
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f'Train size: {train_size}')
    print(f'Val size: {val_size}')
    print(f'Test size: {len(test_dataset)}')
    
    return train_loader, val_loader, test_loader


# ================================================================================
# 5. FUNÇÕES DE TREINAMENTO E AVALIAÇÃO
# ================================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """
    Treina o modelo por uma época
    
    Returns:
        train_loss, train_acc
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Scheduler (se for mini-batch scheduler)
        if scheduler and isinstance(scheduler, (CyclicLR,)):
            scheduler.step()
        
        # Métricas
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc


def evaluate(model, data_loader, criterion, device):
    """
    Avalia o modelo
    
    Returns:
        loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    eval_loss = running_loss / len(data_loader)
    eval_acc = 100. * correct / total
    
    return eval_loss, eval_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, scheduler=None, verbose=True):
    """
    Loop completo de treinamento
    
    Returns:
        history: dicionário com métricas de treinamento
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(num_epochs):
        # Treinar
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        # Avaliar
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Scheduler (se for epoch scheduler)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler and not isinstance(scheduler, (CyclicLR,)):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs} | '
                  f'LR: {current_lr:.6f} | '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
    
    return history


# ================================================================================
# 6. LR RANGE TEST
# ================================================================================

def lr_range_test(model, train_loader, criterion, optimizer, 
                  start_lr=1e-7, end_lr=10, num_iter=100, device='cuda'):
    """
    Learning Rate Range Test
    
    Returns:
        lrs, losses
    """

    model_copy = deepcopy(model)
    model_copy.to(device)
    
    lrs = []
    losses = []
    
    # Calcular multiplicador exponencial
    lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / num_iter)
    temp_scheduler = LambdaLR(optimizer, lr_lambda)
    
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= num_iter:
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward + Backward
        optimizer.zero_grad()
        outputs = model_copy(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        temp_scheduler.step()
        
        # Salvar
        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])
    
    return lrs, losses


# ================================================================================
# 7. VISUALIZAÇÃO
# ================================================================================

def plot_training_history(history, title='Training History'):
    """
    Plota histórico de treinamento
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning Rate
    if 'lr' in history:
        axes[2].plot(history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate')
        axes[2].grid(True)
        axes[2].set_yscale('log')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_lr_range_test(lrs, losses, title='LR Range Test'):
    """
    Plota resultado do LR Range Test
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_gradient_comparison(raw_grads, ewma_grads, adapted_grads, 
                            title='Gradient Comparison'):
    """
    Compara gradientes brutos, suavizados e adaptados
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps = range(len(raw_grads))
    
    # Gradientes brutos
    axes[0].plot(steps, raw_grads, alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Gradient')
    axes[0].set_title('Raw Gradients')
    axes[0].grid(True)
    
    # Gradientes suavizados (EWMA)
    axes[1].plot(steps, ewma_grads, color='red')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Gradient')
    axes[1].set_title('Smoothed Gradients (EWMA)')
    axes[1].grid(True)
    
    # Gradientes adaptados
    axes[2].plot(steps, adapted_grads, color='green')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Gradient')
    axes[2].set_title('Adapted Gradients (Adam)')
    axes[2].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


# ================================================================================
# 8. EXPERIMENTO PRINCIPAL
# ================================================================================

def main():
    """
    Execução principal dos experimentos
    """
    print('='*80)
    print('CAPÍTULO 6 - DEEP LEARNING COM PYTORCH - CIFAR-100')
    print('='*80)
    
    # Configurações
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Preparar dados
    print('\n[1/6] Preparando dados CIFAR-100...')
    train_loader, val_loader, test_loader = prepare_cifar100_data(
        batch_size=BATCH_SIZE,
        val_split=0.1,
        data_augmentation=True
    )
    
    # Criar modelo
    print('\n[2/6] Criando modelo LeNet-5...')
    model = LeNet5_CIFAR100(num_classes=100, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # LR Range Test
    print('\n[3/6] Executando LR Range Test...')
    optimizer_test = optim.Adam(model.parameters(), lr=1e-7)
    lrs, losses = lr_range_test(
        model, train_loader, criterion, optimizer_test,
        start_lr=1e-7, end_lr=1, num_iter=100, device=device
    )
    fig_lr_test = plot_lr_range_test(lrs, losses)
    plt.savefig('lr_range_test.png')
    print('   LR Range Test salvo em: lr_range_test.png')
    
    # Treinar com Adam
    print('\n[4/6] Treinando com Adam...')
    model = LeNet5_CIFAR100(num_classes=100, dropout_p=0.3).to(device)
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    history_adam = train_model(
        model, train_loader, val_loader, criterion, 
        optimizer_adam, NUM_EPOCHS, device, verbose=True
    )
    
    fig_adam = plot_training_history(history_adam, 'Training with Adam')
    plt.savefig('training_adam.png')
    print('   Gráficos Adam salvos em: training_adam.png')
    
    # Avaliar no test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'\n   Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
    
    # Treinar com SGD + Nesterov + Scheduler
    print('\n[5/6] Treinando com SGD + Nesterov + CyclicLR...')
    model_sgd = LeNet5_CIFAR100(num_classes=100, dropout_p=0.3).to(device)
    optimizer_sgd = optim.SGD(
        model_sgd.parameters(), lr=0.01, momentum=0.9, nesterov=True
    )
    scheduler_cyclic = CyclicLR(
        optimizer_sgd, base_lr=0.001, max_lr=0.01,
        step_size_up=len(train_loader), mode='triangular2'
    )
    
    history_sgd = train_model(
        model_sgd, train_loader, val_loader, criterion,
        optimizer_sgd, NUM_EPOCHS, device, scheduler=scheduler_cyclic, verbose=True
    )
    
    fig_sgd = plot_training_history(history_sgd, 'Training with SGD+Nesterov+CyclicLR')
    plt.savefig('training_sgd_nesterov.png')
    print('   Gráficos SGD salvos em: training_sgd_nesterov.png')
    
    # Avaliar no test set
    test_loss_sgd, test_acc_sgd = evaluate(model_sgd, test_loader, criterion, device)
    print(f'\n   Test Loss: {test_loss_sgd:.4f} | Test Acc: {test_acc_sgd:.2f}%')
    
    # Capturar gradientes
    print('\n[6/6] Capturando gradientes...')
    model_grad = LeNet5_CIFAR100(num_classes=100, dropout_p=0.3).to(device)
    optimizer_grad = optim.Adam(model_grad.parameters(), lr=0.001)
    grad_capture = GradientCapture(model_grad)
    grad_capture.register_hooks(['conv1.weight'])
    
    # Treinar por alguns batches para capturar gradientes
    model_grad.train()
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= 100:  # 100 mini-batches
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_grad.zero_grad()
        outputs = model_grad(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_grad.step()
    
    grad_capture.remove_hooks()
    
    # Processar gradientes
    raw_grads = np.array(grad_capture.gradients['conv1.weight']).flatten()[:100]
    ewma_grads = calc_corrected_ewma(raw_grads, beta=0.9)
    
    # Simular adaptação do Adam
    adapted_grads = []
    m = 0
    v = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    for step, g in enumerate(raw_grads, 1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_corrected = m / (1 - beta1 ** step)
        v_corrected = v / (1 - beta2 ** step)
        adapted = m_corrected / (np.sqrt(v_corrected) + eps)
        adapted_grads.append(adapted)
    
    adapted_grads = np.array(adapted_grads)
    
    fig_grads = plot_gradient_comparison(raw_grads, ewma_grads, adapted_grads)
    plt.savefig('gradient_comparison.png')
    print('   Comparação de gradientes salva em: gradient_comparison.png')
    
    print('\n' + '='*80)
    print('EXPERIMENTOS CONCLUÍDOS!')
    print('='*80)
    print('\nRESUMO DOS RESULTADOS:')
    print(f'  Adam: Test Acc = {test_acc:.2f}%')
    print(f'  SGD+Nesterov+CyclicLR: Test Acc = {test_acc_sgd:.2f}%')
    print('\nArquivos gerados:')
    print('  - lr_range_test.png')
    print('  - training_adam.png')
    print('  - training_sgd_nesterov.png')
    print('  - gradient_comparison.png')
    print('='*80)


if __name__ == '__main__':
    main()
