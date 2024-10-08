
# Traffic Signs Recognition using CNN

This project aims to recognize traffic signs using a Convolutional Neural Network (CNN) implemented in Python with the Keras library. Traffic sign recognition is critical for developing autonomous driving systems and enhancing road safety by assisting drivers in recognizing and reacting to traffic signs promptly. Additionally, this project includes integration with a Traffic Management Notification (TMN) system, which sends prediction results via a custom Flask API to facilitate real-time traffic sign recognition and decision-making.

---

<div align="center">
  <img src="./sign.png" alt="Traffic Sign Recognition" style="border:none;">
</div>

---

## Overview

Traffic signs provide essential information and warnings to drivers. However, adverse weather conditions and lack of visibility can lead to accidents. This project leverages deep learning techniques to build a model capable of recognizing traffic signs with high accuracy, thereby contributing to safer driving conditions, especially in autonomous vehicles. The integration with the TMN system allows for real-time application of these predictions, enhancing the potential for immediate action based on the recognized traffic signs.

---

## Convolutional Neural Networks (CNN)

A Convolutional Neural Network is a type of deep learning model particularly effective in image recognition tasks. CNNs automatically and adaptively learn spatial hierarchies of features from input images. They are composed of multiple convolutional layers, which are capable of recognizing various patterns such as edges, textures, and shapes, essential for identifying objects like traffic signs.

---

## Project Contents

- `Traffic-Sign-Recognition.ipynb`: Jupyter notebook containing the code implementation and analysis, including TMN system integration.
- `app.py`: Flask API server code that handles POST requests with prediction data.
- `README.md`: This file, providing an overview of the project.
- `labels.csv`: Label file containing class IDs and corresponding traffic sign names.
  
---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MrMaxMind/Traffic-Signs-Recognition.git
   cd Traffic-Signs-Recognition
2. **Start the Flask API**:
   ```bash
   python app.py
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
4. **Run the Jupyter Notebook**:
   ```bash
   Traffic_Sign_Recognition.ipynb

---

## Libraries and Tools

- `Pandas`: For loading and manipulating data frames.
- `NumPy`: For efficient numerical computations.
- `Matplotlib`: For data visualization.
- `Scikit-learn`: For model evaluation and utilities.
- `OpenCV`: For image processing.
- `TensorFlow`: For building and training the deep learning model.
- `Flask`: For creating the API to integrate the model with a TMN system.
- `Requests`: For sending HTTP requests to the TMN API.


---

## Data Preparation

### Loading the Dataset

The dataset includes images of traffic signs classified into 58 categories. It also contains a `labels.csv` file with class IDs and names.

### Data Visualization

Visualizing random images from the dataset to understand the data.

### Data Preparation for Training

Splitting the dataset into training and validation sets.

---

## Data Augmentation

- To enhance the model's robustness and prevent overfitting, data augmentation techniques are applied.
- Data augmentation methods include random flipping, rotation, and zooming of the images.

---

## Model Architecture

The CNN model consists of:
- Four convolutional layers followed by max-pooling layers.
- Flatten layer to convert 2D arrays to 1D.
- Three fully connected dense layers, with the final layer using softmax activation for classification.

---

## Model Training

- The model is trained using the training set, with early stopping to prevent overfitting.
- Early stopping monitors the validation loss and stops training when the loss stops improving.

---

## Model Evaluation

- The model's performance is evaluated using training and validation accuracy and loss.
- The training process includes plotting the loss and accuracy curves to visualize the model's performance over each epoch.

---

## Traffic Management Notification (TMN) System Integration

After training, the model's predictions are sent to a Traffic Management Notification (TMN) system via a Flask API. This integration enables the model to be used in real-time scenarios, where the predictions can inform traffic management systems to take appropriate actions, enhancing road safety.

### API Setup

The Flask API `app.py` is configured to accept POST requests containing the model's predictions. The predictions are formatted as JSON objects and sent to the API endpoint, where they can be processed and utilized by traffic management systems.

### Example Usage

Once the model is trained, it can make predictions on new images. These predictions are then sent to the TMN system using the `integrate_with_tmn_system()` function within the Jupyter notebook.

---

## Conclusion

- The CNN model demonstrates strong performance in recognizing traffic signs.
- The integration with the TMN system adds a layer of real-time application, making this project valuable for practical deployment in traffic management systems.
- Further enhancements can be made by including more diverse traffic sign images and additional data augmentation techniques.

---

## Contributing

If you have suggestions or improvements, feel free to open an issue or create a pull request.

---

## Acknowledgments
This project is inspired by the need to improve road safety and contribute to the development of autonomous driving technologies.

---

## Thank you for visiting! If you find this project useful, please consider starring the repository. Happy coding!

---
