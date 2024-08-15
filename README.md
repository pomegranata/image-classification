# Image Classification with Jupyter Notebook: Rock, Paper, Scissors
Welcome to the Image Classification project that focuses on classifying images of rock, paper, and scissors. This project was created as a final project for the Dicoding course, Machine Learning for Beginners. It demonstrates how to use machine learning techniques to build an image classifier using Python and Jupyter Notebook. The project leverages convolutional neural networks (CNNs) to automatically learn patterns from image data and classify images into one of the three categories: rock, paper, or scissors.

## Table of Contents
1. [Project Overview]()
2. [Features]()
3. [Dataset]()
4. [Requirements]()
5. [Installation]()
6. [Usage]()
7. [Model Architecture]()
8. [Evaluation]()
9. [Visualization]()
10. [Contributing]()

## Project Overview
The goal of this project is to build a robust image classification model capable of distinguishing between images of rock, paper, and scissors. By employing deep learning techniques, specifically CNNs, the model learns to identify unique features in the images that differentiate one category from another. This project is a practical example of applying machine learning to solve real-world problems and can serve as a foundation for more complex image classification tasks.

## Objectives

1. To explore and preprocess image data for machine learning.

2. To design and train a CNN model using TensorFlow and Keras.

3. To evaluate the model's performance and improve accuracy.

4. To deploy the model for making predictions on new image data.

## Features

1. **Comprehensive Data Preprocessing:** Handles image resizing, normalization, and augmentation to improve model performance.

2. **Flexible Model Architecture:** Utilizes CNNs with configurable layers to experiment with different network designs.

3. **Performance Evaluation:** Includes metrics such as accuracy, precision, recall, and F1-score for thorough evaluation.

4. **Interactive Visualizations:** Provides tools for visualizing model performance and prediction results.

5. **Deployment Ready:** Prepares the model for deployment with options for exporting the trained model.

## Dataset

The dataset for this project consists of images of hands showing rock, paper, and scissors gestures. It can be sourced from [Kaggle's Rock Paper Scissors Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) or other similar repositories. The dataset includes:

1. **Rock Images:** Images depicting the rock gesture.

2. **Paper Images:** Images depicting the paper gesture.

3. **Scissors Images:** Images depicting the scissors gesture.

Each category has a balanced number of images to ensure fair training and evaluation of the model.

## Requirements

To run this project, ensure you have the following software and libraries installed:

* **Python 3.7 or higher:**  The programming language used for this project.

* **Jupyter Notebook:** An interactive environment for running Python code.

* **Split-Folders:** For splitting the dataset into training, validation, and test sets.

* **TensorFlow and Keras:** Libraries for building and training the machine learning model.

* **NumPy:** For numerical computations and array manipulations.

* **Matplotlib:** For plotting and visualizing data and results.

* **Pandas:** For data manipulation and analysis.

## Installation

To set up the project locally, follow these steps:

**1. Clone the repository:**

```
git clone https://github.com/pomegranata/image-classification
cd image-classification
```

**2. Create a virtual environment:**

```
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```

**3. Install the required packages:**

```
pip install -r requirements.txt
```

**4. Launch Jupyter Notebook:**

```
jupyter notebook
```

**5. Open the image-classification.ipynb notebook file.**

## Running on Google Colab
You can also run this project on Google Colab, which provides free access to GPU resources for faster model training:

**1. Upload the notebook:**

* Go to [Google Colab](https://colab.research.google.com/).

* Click on "File" > "Upload Notebook" and select the image-classification.ipynb file from your local system.

**2. Mount Google Drive:**

To access the dataset stored in Google Drive, run the following code cell in Colab:

```
from google.colab import drive
drive.mount('/content/drive')
```

Follow the prompts to authorize access to your Google Drive account.

**3. Install necessary packages:**

Use:

```
!pip install
``` 

To install any additional packages required:

```
!pip install tensorflow
```

**4. Run the notebook:**

Execute the cells in the notebook to preprocess data, train the model, and evaluate performance.

## Usage

To run the project, follow these steps in the Jupyter Notebook or Colab:

**1. Load and explore the dataset:**

* Import images and visualize samples from each category.
* Analyze the distribution of images in the dataset.

**2. Preprocess the data:**

* **Image Resizing:** Resize images to a consistent size for input into the model.

* **Normalization:** Scale pixel values to a range of [0, 1] for better convergence.

* **Data Augmentation:** Apply transformations such as rotation and flipping to enhance the dataset.

**3. Split the data:**

* Use the split-folders library to divide the dataset into training, validation, and test sets:

```
import splitfolders

splitfolders.ratio('/tmp/rockpaperscissors/rps-cv-images', output='/tmp/rockpaperscissors/rps-cv-images', seed=1337, ratio=(0.60,0.40))
```

* This command will split the dataset into 60% training, 40% validation.

**4. Define the CNN model architecture:**

* Configure the layers of the CNN, including convolutional layers, pooling layers, and dense layers.

* Compile the model with an appropriate optimizer and loss function.

**5. Train the model on the training dataset:**

* Monitor training progress and validate the model on the validation set.

**6. Evaluate the model performance on the test dataset:**

* Use metrics such as accuracy and confusion matrix to assess the model's effectiveness.

**7. Use the model to make predictions on new images:**

* Load new images and predict their classes using the trained model.