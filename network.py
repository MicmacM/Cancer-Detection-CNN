import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from BreakHisDataset import BreakHisFiltered
from torch.utils.data import ConcatDataset, random_split, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)


        #self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # 224x224
        # 224 = 2 * 2 * 2 * 4 * 4
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 4)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

def train(model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()  
        pred = outputs.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(target)
    print(f'Train Epoch: {epoch} \tLoss: {loss.item():.6f}\tAccuracy: {accuracy:.2f}%')
    return loss.item(), accuracy


def test(model, device, loss_fn, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            pred = output.argmax(dim=1)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(target)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Affichage du rapport complet (Precision, Recall, F1)
    print("\n--- Rapport de Classification ---")
    print(classification_report(all_targets, all_preds, target_names=['Normal', 'Anormal']))
    
    # Affichage de la matrice de confusion
    print("--- Matrice de Confusion ---")
    print(confusion_matrix(all_targets, all_preds))
    return loss.item(), accuracy
    
def main():

    target_subtypes_benign = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
    target_subtypes_malignant = ['lobular_carcinoma', 'ductal_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
    transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(),   
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    benign_dataset = BreakHisFiltered(
        root_dir='data/benign/SOB/', 
        transform=transform_pipeline,
        magnification=['40','100'],
        subtypes=target_subtypes_benign,
        label=0
    )

    malignant_dataset = BreakHisFiltered(
        root_dir='data/malignant/SOB/', 
        transform=transform_pipeline,
        magnification=['40','100'],
        subtypes=target_subtypes_malignant,
        label=1
    )

    full_dataset = ConcatDataset([benign_dataset, malignant_dataset])

    # We use 80% of the data for training and 20% for testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_set, test_set = random_split(full_dataset, [train_size, test_size], 
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    print(f"Total: {len(full_dataset)} | Train: {len(train_set)} | Test: {len(test_set)}")



    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    weights = torch.tensor([2.21, 1.0]).to(device)
    loss_fn = nn.NLLLoss(weight=weights)
    epochs = 10

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    best_accuracy = 0.0
    for epoch in tqdm(range(epochs), "Epoch : "):
        train_loss, train_acc = train(model, device, train_loader, optimizer, loss_fn, epoch)
        test_loss, test_acc = test(model, device, loss_fn, test_loader)
        #scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'cnn_v1.pth')


    
    plt.figure()
    # Graphique de la Loss
    plt.subplot(1, 2, 1)
    plt.plot(list(range(len(train_losses))), train_losses, 'b-', label='Train Loss')
    plt.plot(list(range(len(test_losses))), test_losses, 'r-', label='Test Loss')
    plt.title('Loss each epoch')
    plt.legend()
    
    # Graphique de l'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(train_accs))), train_accs, 'b-', label='Train Acc')
    plt.plot(list(range(len(test_accs))), test_accs, 'r-', label='Test Acc')
    plt.title('Accuracy each epoch')
    plt.legend()
    
    plt.show()

    


if __name__ == '__main__':
    main()