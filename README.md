# 🎾 Human Activity Recognition System
> **An AI-powered web application for real-time tennis swing classification.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 Project Overview
This project focuses on identifying athletic movements in tennis using **Deep Learning** and **Computer Vision**. By leveraging a custom-trained CNN architecture, the system can analyze video input and provide high-confidence predictions for specific actions like serves, forehands, and backhands.

### Key Features
* **High Accuracy:** Currently achieving **97.29%** confidence on action classification.
* **Real-time Interface:** Built with **Streamlit** for seamless user interaction and video upload.
* **Optimized Pipeline:** Pre-processing and inference optimized for low-latency performance.

---

## 🚀 Live Demo
![App Screenshot](app_screenshot.png)
*Figure 1: Streamlit interface predicting a "Tennis Swing" with 97.29% confidence.*

---

## 🛠️ Tech Stack
* **Frameworks:** PyTorch, Torchvision
* **Computer Vision:** OpenCV (cv2)
* **Web App:** Streamlit
* **Analysis:** NumPy, Matplotlib, Pandas

---

## 🧬 Model Details
The underlying model uses a spatial-temporal approach to understand movement.
* **Architecture:** CNN-based feature extraction.
* **Loss Function:** Cross-Entropy.
* **Optimizer:** Adam.
* **Training Platform:** Trained using GPU acceleration for rapid convergence.

---

## 💻 Getting Started

### Prerequisites
* Python 3.9+
* pip (Python package manager)

### Installation
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YousefAbdalbary/tennis-action-recognition.git](https://github.com/YousefAbdalbary/tennis-action-recognition.git)
   cd tennis-action-recognition
