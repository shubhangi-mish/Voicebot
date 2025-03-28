# Voice AI Chatbot

This is a Voice AI Chatbot that allows users to interact using voice input and receive spoken responses in real-time. The application uses:

Whisper ASR (for speech-to-text transcription)

Google Gemini AI (for generating AI responses)

pyttsx3 (for text-to-speech conversion)

Gradio (for building the web interface)

## 🚀 Installation Guide

1️⃣ Clone the Repository

git clone https://github.com/your-repo-name.git
cd your-repo-name

2️⃣ Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

Make sure you have ffmpeg installed for audio processing.

sudo apt install ffmpeg  # Ubuntu/Linux
brew install ffmpeg      # macOS

## 🔑 API Keys Setup

This project requires API keys for Hugging Face (Whisper ASR) and Google Gemini AI.

Hugging Face Token: Get a token from Hugging Face and set it in a .env file.

Google Gemini API Key: Get a key from Google AI and add it to .env.

Create a .env file in the project root:

HF_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_google_api_key

## 🎤 Running the Application

Run the Web App

python gui.py

This will start the Gradio web interface. Open the provided URL in your browser (http://127.0.0.1:7860).

## 🌍 Deploying the Web App

You can deploy the application on Hugging Face Spaces, Render, or Google Cloud.

Deploy on Hugging Face Spaces

Push your repository to Hugging Face.

In the Hugging Face app.py, use iface.launch(share=True) to expose the app publicly.

## 📌 Features

✅ Voice Input & Output✅ Real-time AI Chat✅ Powered by Whisper & Gemini AI✅ Web-based Interface using Gradio✅ Easily Deployable on Cloud Platforms

## 🤖 Future Enhancements

Support for multiple languages

Improved response speed

Additional chatbot functionalities