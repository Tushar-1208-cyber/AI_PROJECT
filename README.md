# 🖼️ AI-Powered Image Classification System

A web application that classifies uploaded images in real-time using a deep learning model, with a responsive frontend connected to a Python backend via REST APIs.

## 🎯 Overview

This project lets a user upload an image through a web interface and get an instant prediction of what the image contains. It combines a pretrained deep learning model (transfer learning) with a lightweight backend and a clean frontend, making it a complete end-to-end ML web application rather than just a notebook experiment.

## 🛠️ Tech Stack

- **Model:** TensorFlow, Keras, EfficientNetB0 (pretrained CNN, fine-tuned for this classification task)
- **Backend:** Python, REST APIs (serves predictions to the frontend)
- **Frontend:** HTML, CSS, JavaScript
- **Core Language Split:** JavaScript, HTML, CSS (UI) + Python (model & backend logic)

## ✨ Features

- Upload any image directly from the browser
- Real-time classification using EfficientNetB0
- REST API layer connecting frontend and backend cleanly
- Responsive, simple UI for quick testing

## 🧠 How It Works

1. **Frontend** — User selects/uploads an image through the HTML/CSS/JS interface.
2. **REST API call** — The image is sent from the frontend to the Python backend via an API request.
3. **Model Inference** — The backend loads the EfficientNetB0 model (pretrained on ImageNet and adapted for this task) and runs prediction on the uploaded image.
4. **Response** — The predicted class (and confidence score) is sent back and displayed instantly on the frontend.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Tushar-1208-cyber/AI_PROJECT.git
cd AI_PROJECT

# Install Python dependencies
pip install tensorflow keras flask numpy pillow

# Run the backend server
python app.py
```

Then open the frontend HTML file in your browser (or navigate to the local server URL shown in the terminal) and upload an image to test the classifier.

## 📂 Project Structure

```
AI_PROJECT/
├── AI_PROJECT/          # Core application files (frontend + backend)
├── .vscode/             # Editor config
└── README.md
```

## 📈 What I Learned

- Applying transfer learning with EfficientNetB0 instead of training a CNN from scratch
- Connecting a Python ML backend to a JavaScript frontend using REST APIs
- Handling real-time image upload and inference workflows end-to-end

## 🔮 Future Improvements

- Add multi-class confidence visualization (bar chart of top predictions)
- Deploy the app using Docker for easier setup
- Add automated tests for the API endpoints
