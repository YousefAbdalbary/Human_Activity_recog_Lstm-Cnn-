import streamlit as st
import tempfile
import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# ==========================================
# 1. THE MODEL ARCHITECTURE
# ==========================================
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM, self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(
            input_size=1280, hidden_size=64, num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        b, seq, c, h, w = x.size()
        x = x.view(b * seq, c, h, w)
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = self.pool(x)
        x = x.view(b * seq, -1)
        x = x.view(b, seq, -1)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)


# ==========================================
# 2. CACHED MODEL LOADING
# ==========================================
# This decorator ensures the model is only loaded into memory ONCE
@st.cache_resource
def load_model(model_path="cnn_lstm_action_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        return None, None, None

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]

    model = CNN_LSTM(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, classes, device


# ==========================================
# 3. VIDEO PROCESSING
# ==========================================
def process_video(video_path, sequence_length=20, image_size=128):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_interval = max(int(frame_count / sequence_length), 1)

    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame)
        frames.append(frame_tensor)

    cap.release()

    while len(frames) < sequence_length:
        frames.append(torch.zeros((3, image_size, image_size)))

    return torch.stack(frames)


# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(
    page_title="Action Recognition AI", page_icon="🎥", layout="centered"
)

st.title("🎥 Human Activity Recognition")
st.markdown("Upload a video and the CNN-LSTM model will predict the action!")

# Try to load the model
model, classes, device = load_model()

if model is None:
    st.error(
        "❌ Model file `cnn_lstm_action_model.pth` not found! Please place it in the same folder as this script."
    )
    st.stop()

st.sidebar.success(f"**Model Loaded Successfully on {str(device).upper()}!**")
st.sidebar.markdown("**Trained Classes:**")
for c in classes:
    st.sidebar.markdown(f"- {c}")

# File Uploader
uploaded_file = st.file_uploader("Upload a sports video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Display the video in the app
    st.video(uploaded_file)

    if st.button("Predict Action", use_container_width=True):
        with st.spinner("Extracting frames and analyzing..."):

            # OpenCV requires a physical file path, so we save the upload temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()

            try:
                # 1. Process Video
                frames_tensor = process_video(tfile.name)
                input_tensor = frames_tensor.unsqueeze(0).to(device)

                # 2. Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]

                pred_idx = torch.argmax(probabilities).item()
                pred_class = classes[pred_idx]
                confidence = probabilities[pred_idx].item() * 100

                # 3. Display Results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")

                col1, col2 = st.columns(2)
                col1.metric("Action Detected", pred_class)
                col2.metric("Confidence", f"{confidence:.2f}%")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

            finally:
                # Clean up the temporary file
                os.unlink(tfile.name)
