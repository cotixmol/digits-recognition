# 🧠 Digit Recognition with CNN & OpenCV

Welcome to the **Digit Recognition** repository! This project uses a Convolutional Neural Network (CNN) and OpenCV to recognize handwritten digits in real-time using your webcam. 📷✨

## 🚀 Getting Started

Follow the steps below to get everything running smoothly.

### 1. 🏋️‍♂️ Train the Model

Before you can predict digits, you'll need to train the model on the MNIST dataset. Simply run the following script:

```bash
python train_model.py
```

This will train a CNN on the MNIST dataset and save the model for later use.

### 2. 🎥 Predict Digits Using Your Webcam

After training the model, you can test it by running the webcam prediction script:

```bash
python webcam_predict.py
```

Your webcam will activate, and the model will start predicting the digits you draw or show in front of the camera!

## 🔧 Modifying the Model

If you'd like to tweak the model architecture, head over to the `cnn_model.py` file. In this file, you can modify the layers of the CNN to suit your needs. 🤓

```bash
cnn_model.py
```

Feel free to experiment with different architectures or hyperparameters to improve accuracy or performance.

## 📂 Project Structure

- `train_model.py` - Script to train the CNN on the MNIST dataset.
- `webcam_predict.py` - Script to capture webcam input and predict digits in real-time.
- `cnn_model.py` - The CNN architecture used for digit recognition.
- `data_loader.py` - The MNIST data import
- `README.md` - You're reading it! 😄

## 📋 Requirements

Make sure you have all the required libraries installed:

```bash
pip install -r requirements.txt
```

## 💡 Notes

- Make sure your webcam is working properly before running `webcam_predict.py`.
- If you want to train the model again from scratch, simply re-run `train_model.py`.

Happy coding and enjoy predicting digits! 🔢🎉
