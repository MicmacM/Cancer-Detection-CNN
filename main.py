import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.optim.lr_scheduler import StepLR
from BreakHisDataset import BreakHisDataset
from torch.utils.data import ConcatDataset, random_split, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from CustomCNN import Net



parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Name of the model : custom / resnet')
parser.add_argument('--gpu', default="mps", type=str, help='Name of the GPU : cuda/mps')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
parser.add_argument('--mag', default='40', type=str, help="Magnification to use : 40,100,200,400")
parser.add_argument('--batch_size', default=64, type=int, help="Batch size for training")
args = parser.parse_args()

# SET GPU
if args.gpu == "mps":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
elif args.gpu == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


def train(model, device, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = len(train_loader.dataset)

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()  
        _,preds = torch.max(outputs.data,1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == target.data)
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.data.item() / total_samples
    return epoch_loss, epoch_acc


def test(model, device, loss_fn, test_loader):
    model.eval()
    total_samples = len(test_loader.dataset)
    predictions = np.zeros(total_samples)
    all_targets = np.zeros(total_samples)
    all_proba = np.zeros((total_samples,2))
    i = 0
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            _,preds = torch.max(output.data,1)
            # statistics
            running_loss += loss.data.item() * input.size(0)
            running_corrects += torch.sum(preds == target.data)

            batch_len = len(target)
            predictions[i:i+batch_len] = preds.cpu().numpy()
            all_targets[i:i+batch_len] = target.cpu().numpy()
            all_proba[i:i+batch_len, :] = output.cpu().numpy()
            i += batch_len
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.data.item() / total_samples
    print('Loss: {:.4f} Acc: {:.4f}'.format(
                     epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc, predictions, all_targets



if __name__ == '__main__':
    # LOAD DATA
    if args.mag:
        magnification = [int(i) for i in list(args.mag.split(','))]

    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)), 
        #transforms.RandomHorizontalFlip(), 
        #transforms.RandomVerticalFlip(),   
        #transforms.RandomRotation(90),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BreakHisDataset(
        data_folder_path='data/', 
        magnification=magnification,
        transform=transform_pipeline,
        train=True
    )
    test_dataset = BreakHisDataset(
        data_folder_path='data/', 
        magnification=magnification,
        transform=transform_pipeline, 
        train=False
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,      
        num_workers=4      
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,    
        num_workers=4
    )

    if args.epochs:
        epochs = args.epochs
    if args.lr:
        learning_rate = args.lr
    if args.batch_size:
        batch_size = args.batch_size

    if args.model == 'custom':
        model = Net.to(device)


        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.NLLLoss()
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    elif args.model == 'resnet':
        base_model = models.resnet50(weights='DEFAULT').to(device)
        base_model.eval()

        # Custom feature extractor
        feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        def preconvfeat(dataloader):
            feats, labs = [], []
            with torch.no_grad():
                for img, label in dataloader:
                    img = img.to(device)
                    out = feature_extractor(img) # Output is [batch, 2048, 1, 1]
                    out = out.view(out.size(0), -1) # Flatten to [batch, 2048]
                    feats.extend(out.cpu().numpy())
                    labs.extend(label.numpy())
            return np.array(feats), np.array(labs)
        

        # Preconvolute features for transfer learning
        conv_feat_train,labels_train = preconvfeat(train_loader)
        conv_feat_test,labels_test = preconvfeat(test_loader)
        print("Preconvolution done.")

        # Create new loaders with conv features
        train_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(conv_feat_train).float(), torch.from_numpy(labels_train).long()),
        batch_size=128, shuffle=True
        )
        test_loader = DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(conv_feat_test).float(), torch.from_numpy(labels_test).long()),
            batch_size=128, shuffle=False
        )

        model = nn.Sequential(
            nn.Linear(2048, 2),
            nn.LogSoftmax(dim=1)
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.NLLLoss()
        
    # TRAINING LOOP :
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_accuracy = 0.0

    for epoch in tqdm(range(epochs), desc="Epoch"):
        train_loss, train_acc = train(model, device, train_loader, optimizer, loss_fn)
        test_loss, test_acc, predictions, all_targets = test(model, device, loss_fn, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"Best Test Accuracy: {best_accuracy:.2f}%")

    plt.figure()
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(list(range(len(train_losses))), train_losses, 'b-', label='Train Loss')
    plt.plot(list(range(len(test_losses))), test_losses, 'r-', label='Test Loss')
    plt.title('Loss each epoch')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(train_accs))), train_accs, 'b-', label='Train Acc')
    plt.plot(list(range(len(test_accs))), test_accs, 'r-', label='Test Acc')
    plt.title('Accuracy each epoch')
    plt.legend()

    plt.show()

