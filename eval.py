import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import ConcatDataset, random_split, DataLoader
from BreakHisDataset import BreakHisFiltered
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- 1. Architecture du Réseau (Doit être identique à ton script d'entraînement) ---
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
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 256) 
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 4)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# --- 2. Ta fonction test() d'origine ---
def test(model, device, loss_fn, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    
    print("\n--- Rapport de Classification ---")
    print(classification_report(all_targets, all_preds, target_names=['Normal', 'Anormal']))
    
    print("--- Matrice de Confusion ---")
    print(confusion_matrix(all_targets, all_preds))
    
    return avg_loss

def main():

    target_subtypes_benign = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
    target_subtypes_malignant = ['lobular_carcinoma', 'ductal_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
    transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)), 
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
    weights = torch.tensor([1.5, 1.0]).to(device)
    loss_fn = nn.NLLLoss(weight=weights)

    test_loss, test_acc = test(model, device, loss_fn, test_loader)


    

if __name__ == '__main__':
    main()