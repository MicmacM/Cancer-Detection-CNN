# Cancer Detection using CNN

This repository contains the implementation of a Convolutional Neural Network (CNN) for binary classification of cancer tissue images using the BreakHis dataset.

There is also the possibility of doing transfer learning from the ResNet50 model on the same dataset.

## Dataset

The [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis) is a publicly available dataset containing histopathological images of benign and malignant tumors. It is widely used for research in cancer detection and classification.

<img src="images/benign.png" alt="isolated" width="200"/><img src="images/malignant.png" alt="isolated" width="200"/>

Above is an example of a benign image (left) and malignant image (right)

## Project Overview


- **Objective**: To classify histopathological images as benign or malignant.
- **Model 1**: A custom CNN architecture designed for binary classification. The aim was to experiment with depth, dropout layer, pooling layers, so I choose not to do transfer learning even though that would probably yields better results
- **Model 2**: The ResNet50 model [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights)](Resnet), trained on ImageNet. The aim was to experiment with transfer learning. 

## Architecture for the custom model

I choosed to put 4 convolutional layers with increasing number of kernel : $8,32,64,128$


#### Taking the slides colors into account
Due to the data images being from microscope slides images, there is a different dosage of colorant in them. To take that into account and preventing the model from learning of theses different shades, I used a transformation
`transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` that randomly changes the brightness, contrast, saturation and hue of an image.

#### Preventing overfitting

##### Dropout layer
##### 

## ResNet model

In order to speed up the learning process, I used a preconvolution of the features.

The downsides of this approach is that we can't use the various data augmentation techniques used by the custom method, such as Color Jitter to take the microscope slide colorant into account, or the orientation of the images, but because ResNet is a way deeper model, it is able to learn even without that.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/MicMacM/Cancer-Detection-CNN.git
    cd Cancer-Detection-CNN
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and prepare the BreakHis dataset:
    - Download the dataset on Kaggle : to download the dataset.
    - Place the dataset in the `data/` directory.

## Usage

```bash
python main.py --model "resnet" --gpu "cuda" --epochs 50 --lr 0.0001 --mag "40,100,400" --batch-size 128
```

Args : 

`--model : "resnet" | "custom"` : Model to use
`--gpu : "cuda" | "mps"` : Use GPU if availible else CPU. (`mps` if for Mac Silicon)
`--mag : "40, 100, 200, 400"` : Magnification available for the images



## Results

### CustomCNN

###

## License

This project is licensed under the [MIT License](LICENSE).
