# Cancer Detection using CNN

This repository contains the implementation of a Convolutional Neural Network (CNN) for binary classification of cancer tissue images using the BreakHis dataset.

## Dataset

The [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis) is a publicly available dataset containing histopathological images of benign and malignant tumors. It is widely used for research in cancer detection and classification.

<img src="images/benign.png" alt="isolated" width="200"/><img src="images/malignant.png" alt="isolated" width="200"/>

Above is an example of a benign image (left) and malignant image (right)

## Project Overview


- **Objective**: To classify histopathological images as benign or malignant.
- **Model**: A custom CNN architecture designed for binary classification. The aim was to experiment with depth, dropout layer, pooling layers, so I choose not to do transfer learning even though that would probably yields better results

## Architecture

I choosed to put 4 convolutional layers with increasing number of kernel : $8,32,64,128$


### Taking the slides colors into account
Due to the data images being from microscope slides images, there is a different dosage of colorant in them. To take that into account and preventing the model from learning of theses different shades, I used a transformation
`transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` that randomly changes the brightness, contrast, saturation and hue of an image.

### Preventing overfitting

#### Dropout layer
#### 



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

1. Train the model:
    ```bash
    python train.py
    ```

2. Evaluate the model:
    ```bash
    python evaluate.py
    ```

3. Make predictions:
    ```bash
    python predict.py --image path/to/image.jpg
    ```

## Results

## License

This project is licensed under the [MIT License](LICENSE).
