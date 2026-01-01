import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from torchvision import transforms

class BreakHisFiltered(Dataset):
    def __init__(self, root_dir, transform=None, magnification='40', subtypes=None, label = 0):
        """
        Custom Dataset for BreakHis dataset
        
        root_dir: path to the root directory containing images
        transform: (torch.Transform) transformation to apply
        magnification: list(str) : list of magnifications (ex : ['40', '100'])
        subtypes: list(str) : list of subtypes (subfolders) to include
        label: (int) Label of the class (0 or 1)
        """
        self.transform = transform
        self.image_paths = []
        self.label = label
        
        # Get all png files
        all_files = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        
        
        for path in all_files:
            filename = os.path.basename(path)
            
            # Magnification (ex: "-40-")
            mag_in_file = False
            for mag in magnification:
                if f"-{mag}-" in filename:
                    mag_in_file = True
                    break
            if not mag_in_file:
                continue
                
            # ex: subtypes=['adenosis', 'fibroadenoma', ...]
            if subtypes:
                if not any(st.lower() in path.lower() for st in subtypes):
                    continue
            
            self.image_paths.append(path)
            
        print(f"Loading complete : {len(self.image_paths)} images found for {magnification}X.")

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx])

        image = image.float() / 255.0

        if self.transform:
            image = self.transform(image)
            
        return image, self.label

if __name__ == "__main__":
    # List of types to include
    target_subtypes_benign = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
    target_subtypes_malignant = ['lobular_carcinoma', 'ductal_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']

    dataset = BreakHisFiltered(
        root_dir='data/malignant/SOB/', 
        transform=transforms.ToTensor(),
        magnification='40',
        subtypes=target_subtypes_malignant,
        label=1
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)