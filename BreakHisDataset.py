from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
from pathlib import Path

class BreakHisDataset(Dataset):
    def __init__(self, data_folder_path, labels_path="Folds.csv", magnification=[40], transform=None, train=True):
        self.data_folder_path = data_folder_path
        self.transform = transform
        
        df = pd.read_csv(labels_path)
        # Quick change because I changed the structure, remove if necessary
        df['filename'] = df['filename'].str.replace('BreaKHis_v1/histology_slides/breast/', self.data_folder_path, regex=False)
        
        target_grp = 'train' if train else 'test'
        
        mask = (df['grp'] == target_grp) & (df['mag'].isin(magnification))
        self.df = df[mask].copy()
        
        self.df['label'] = self.df['filename'].apply(lambda x: 0 if 'benign' in x.lower() else 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = str(row['filename'])
        
        image = read_image(img_path).float() / 255.0

        if self.transform:
            image = self.transform(image)
            
        return image, row['label']


if __name__ == "__main__":

    # EXAMPLE USAGE
    transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BreakHisDataset(
        data_folder_path='data/', 
        magnification=[40, 100],
        transform=transform_pipeline,
        train=True
    )

    test_dataset = BreakHisDataset(
        data_folder_path='data/', 
        magnification=[40, 100],
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