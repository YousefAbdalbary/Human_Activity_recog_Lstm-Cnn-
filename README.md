
# 🏃 Human Activity Recognition (HAR)
> **A Hybrid Deep Learning Approach using LRCN (CNN-LSTM) for Video Classification.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview
This repository contains a robust pipeline for **Human Activity Recognition (HAR)**. Unlike standard image classification, this project focuses on the temporal aspect of human movement by processing sequences of video frames to identify actions like walking, running, or sports-specific movements.

### Key Highlights
* **Architecture:** Utilizes **LRCN (Long-term Recurrent Convolutional Network)**.
* **Feature Extraction:** CNN layers (TimeDistributed) extract spatial features from each frame.
* **Sequence Learning:** LSTM layers process the temporal flow to understand the action over time.
* **High Confidence:** Capable of achieving high accuracy (e.g., 97%+) on specialized action datasets.

---

## 🚀 Live Demo & Visuals
![App Demo](app_screenshot.png)
*Figure 1: Streamlit interface predicting a "Tennis Swing" with 97.29% confidence.*

---

## 🧬 Model Architecture
The model is designed to handle the complexity of video data by combining two powerful neural networks:

1.  **Convolutional Neural Network (CNN):** A pre-trained or custom CNN processes individual frames to identify objects and body positions.
2.  **Long Short-Term Memory (LSTM):** A recurrent layer that "remembers" the previous frames to classify the movement as a continuous action.

**Data Flow:**
`Input Video` ➡️ `Frame Extraction` ➡️ `CNN (Spatial Features)` ➡️ `LSTM (Temporal Features)` ➡️ `Softmax Classification`

---

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Deployment/UI:** Streamlit
* **Data Science:** NumPy, Pandas, Matplotlib

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YousefAbdalbary/Human_Activity_recog_Lstm-Cnn-.git](https://github.com/YousefAbdalbary/Human_Activity_recog_Lstm-Cnn-.git)
cd Human_Activity_recog_Lstm-Cnn-
````

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3\. Run Inference

To test the model on your own video files:

```bash
python predict.py --video path/to/your/video.mp4
```

-----

## 📊 Performance

| Activity | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Tennis Swing | 0.98 | 0.97 | 0.97 |
| Walking | 0.94 | 0.92 | 0.93 |
| Running | 0.91 | 0.95 | 0.93 |

-----

## 👤 Author

**Yousef Abdalbary**
*AI Engineer*

  * [GitHub](https://www.google.com/search?q=https://github.com/YousefAbdalbary)
  * [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/yousef-abdalbary)


```
```
