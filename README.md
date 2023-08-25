# Face Recognition with LenetModel

This repository contains the project for face recognition using a Convolutional Neural Network and OpenCV. The project includes two main scripts: main.ipynb and main.py. The first script trains a model for face recognition using TensorFlow and Keras, while the second script uses the trained model for real-time face recognition through a webcam feed.

## Prerequisites

Before running the scripts, make sure you have the following prerequisites installed:

- Python 3.x
- TensorFlow (`pip install tensorflow`)
- Keras (`pip install keras`)
- OpenCV (`pip install opencv-python`)
- Matplotlib (`pip install matplotlib`)
- NumPy (`pip install numpy`)
- scikit-learn (`pip install scikit-learn`)
- seaborn (`pip install seaborn`)
- albumentations (`pip install albumentations`)
- tensorflow_datasets (`pip install tensorflow-datasets`)
- tensorflow_probability (`pip install tensorflow-probability`)

## Folder Structure

- `Code`: This directory contains the Python scripts for the Lenet model training (`main.ipynb`) and real-time face recognition (`main.py`).
- `Famous Person Dataset`: This directory contains the training and validation data for the face recognition model.
- `Model`: This directory contains the trained Lenet model (`Face Recognizer.h5`).

## Usage

### main.ipynb

This script trains a Lenet model for face recognition using the TensorFlow framework. It uses the provided training and validation datasets to build and train the model.

To run the script:

1. Organize your data in the "Famous Person Dataset" directory with "train" and "valid" subdirectories containing images.
2. Configure the script's parameters in the `CONFIGURATION` dictionary.
3. Run the notebook step by step.

### main.py

This script performs real-time face recognition using the trained Lenet model and OpenCV's Haarcascade classifier. It captures video from the default camera and detects faces, then uses the Lenet model to predict the recognized person.

To run the script:

1. Place the trained Lenet model (`Face Recognizer.h5`) in the "Model" directory.
2. Run the `main.py` script.

## Contributing

Contributions to this repository are welcome! If you find any issues or have suggestions for improvements, please feel free to submit a pull request.
