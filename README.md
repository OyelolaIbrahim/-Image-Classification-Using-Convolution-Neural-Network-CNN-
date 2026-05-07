# Image Classification Using Convolutional Neural Network (CNN)

Classifying 60,000 colour images across 10 object categories 
using a custom CNN architecture trained on the CIFAR-10 dataset 
with Keras and TensorFlow.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-Keras%20%2F%20TensorFlow-orange)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![Type](https://img.shields.io/badge/Type-Multi--Class%20Classification-blue)

---

## Overview

Builds and trains a Convolutional Neural Network (CNN) from scratch 
using Keras to classify 32×32 colour images from the CIFAR-10 
dataset into 10 categories: airplane, automobile, bird, cat, deer, 
dog, frog, horse, ship, and truck. The pipeline covers data 
normalisation, custom CNN architecture design with Dropout 
regularisation, 200-epoch training, and full evaluation using 
a classification report and annotated confusion matrix heatmap.

---

## Problem Statement

Image classification — automatically assigning a label to an image 
from a fixed set of categories — is a foundational computer vision 
task. CIFAR-10 is one of the most widely used benchmarks for 
evaluating CNN architectures because its 10 classes include both 
similar-looking categories (e.g. cat vs dog, automobile vs truck) 
that challenge the model to learn fine-grained visual features.

---

## Dataset

- **Name:** CIFAR-10 Image Classification Dataset
- **Source:** Built into Keras — loaded automatically via 
  `keras.datasets.cifar10.load_data()` — no download needed
- **Size:** 60,000 colour images
  - Training: 50,000 images
  - Test: 10,000 images
- **Image Size:** 32×32 pixels, RGB (3 colour channels)
- **Classes:** 10 categories —
  `airplane, automobile, bird, cat, deer, 
   dog, frog, horse, ship, truck`
- **Format:** NumPy arrays loaded directly from Keras

> No dataset download required. Running the notebook will 
> automatically download CIFAR-10 the first time.

---

## Approach

**Data Preprocessing**
- Loaded CIFAR-10 using `keras.datasets.cifar10.load_data()`
- Visualised 25 sample test images in a 5×5 grid with 
  class name labels using Matplotlib
- Converted image arrays to `float32` for numerical stability
- Normalised pixel values using mean subtraction and 
  max scaling:
```python
  X_train = (X_train - X_train.mean()) / X_train.max()
  X_test  = (X_test  - X_test.mean())  / X_test.max()
```
- One-hot encoded labels using 
  `keras.utils.to_categorical(y, 10)`

**CNN Architecture**

| Layer | Type | Details |
|-------|------|---------|
| 1 | Conv2D | 32 filters, 3×3 kernel, same padding, ReLU |
| 2 | MaxPooling2D | 2×2 pool size |
| 3 | Conv2D | 64 filters, 3×3 kernel, same padding, ReLU |
| 4 | MaxPooling2D | 2×2 pool size |
| 5 | Dropout | Rate = 0.25 |
| 6 | Flatten | Converts feature maps to 1D vector |
| 7 | Dense | 256 units, ReLU activation |
| 8 | Dropout | Rate = 0.50 |
| 9 | Dense (output) | 10 units, Softmax activation |

```python
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', 
                 activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```

**Training Configuration**

| Parameter | Value |
|-----------|-------|
| Loss Function | Categorical Cross-Entropy |
| Optimiser | Adam |
| Learning Rate | 1.0e-4 (0.0001) |
| Batch Size | 256 |
| Epochs | 200 |

**Evaluation**
- Generated predictions using 
  `np.argmax(model.predict(X_test), axis=1)`
- Printed full `classification_report` showing Precision, 
  Recall, and F1-Score per class
- Computed overall accuracy using `accuracy_score()`
- Visualised results using an annotated confusion matrix 
  heatmap with class name labels on both axes



## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/OyelolaIbrahim/image-classification-cnn-cifar10.git
cd image-classification-cnn-cifar10

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
#    CIFAR-10 will be downloaded automatically on first run
jupyter notebook image_classification_cnn.ipynb
```

> **Note:** Training for 200 epochs is compute-intensive. 
> A GPU (Google Colab or local CUDA) is strongly recommended. 
> On CPU, training may take several hours.

