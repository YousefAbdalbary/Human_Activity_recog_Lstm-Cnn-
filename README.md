
# 🎥 Human Activity Recognition using CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)



## 📝 Project Overview
This project implements a Deep Learning solution for **Human Activity Recognition (HAR)** using a hybrid **CNN-LSTM** architecture. By combining the spatial feature extraction power of Convolutional Neural Networks with the temporal sequencing capabilities of Long Short-Term Memory networks, the model accurately classifies sports actions from raw video data.

### 🎯 Target Actions
The model is optimized to recognize:
* **Basketball** 🏀
* **CricketShot** 🏏
* **TennisSwing** 🎾

---

## 🏗️ Architecture Details
The model utilizes **Transfer Learning** to achieve high accuracy with minimal training time.

1.  **Spatial Feature Extractor (CNN):** A frozen **MobileNetV2** backbone (pretrained on ImageNet) processes 20 frames per video to extract 1280 essential visual features.
2.  **Temporal Analyzer (LSTM):** A 64-unit **LSTM** layer interprets the sequence of these features to understand the motion.
3.  **Classifier:** A Fully Connected layer with **Softmax** activation outputs the final probability.



### Model Specifications
| Feature | Detail |
| :--- | :--- |
| **Base Model** | MobileNetV2 (Frozen) |
| **Recurrent Layer** | LSTM (64 Hidden Units) |
| **Sequence Length** | 20 Frames |
| **Total Params** | 2,568,643 |
| **Trainable Params** | 344,771 |
| **Accuracy** | **98.67% (Validation)** |

---

## 📊 Training Performance
The model was trained for 10 epochs on a curated subset of the **UCF101** dataset. 

| Epoch | Loss | Accuracy (%) |
| :--- | :--- | :--- |
| 1 | 0.8822 | 59.73% |
| 5 | 0.3300 | 90.38% |
| 10 | 0.1605 | 95.30% |

---

## 🚀 How to Run

### 1. Requirements
Ensure you have Python installed, then run:
```bash
pip install torch torchvision opencv-python pandas numpy streamlit
````

### 2\. Launch the Web App

The project includes a **Streamlit** interface for easy testing. Run the following command in your terminal:

```bash
streamlit run app.py
```

### 3\. Usage

1.  Open the local URL provided by Streamlit.
2.  Upload a `.mp4` or `.avi` video of Basketball, Cricket, or Tennis.
3.  Click **Predict Action** to see the results.

-----

## 🖼️ Application Preview

Below is a screenshot of the system successfully identifying a Tennis Swing with high confidence:

-----

## 📂 Project Structure

```text
.
├── dataset_root/          # Preprocessed video data
├── app.py                 # Streamlit Web Application
├── train_model.py         # Model training script
├── cnn_lstm_action_model.pth # Saved model weights
└── README.md              # Project documentation
```

```

