import os
import torch
import sounddevice as sd
import numpy as np
import torchaudio
import pyttsx3
import threading
import tkinter as tk
from tkinter import scrolledtext
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not HF_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Missing API tokens. Please set HF_TOKEN and GEMINI_API_KEY in your .env file.")

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper model
model_id = "openai/whisper-medium.en"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True, token=HF_TOKEN
).to(device)
processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Configure Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)
engine = pyttsx3.init()

# Load personal information
def load_personal_info(filename="about_me.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except:
        return ""

personal_info = load_personal_info()

# Function to convert text to speech
def speak_response(text):
    engine.say(text)
    engine.runAndWait()

# Function to record user audio
def record_audio(duration=5, samplerate=16000, filename="input.wav"):
    try:
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        write(filename, samplerate, audio)
        return filename
    except:
        return None

# Function to transcribe recorded audio
def transcribe_audio(filename):
    try:
        waveform, sample_rate = torchaudio.load(filename)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        audio_data = {"array": waveform.squeeze(0).numpy(), "sampling_rate": 16000}
        return asr_pipeline(audio_data)["text"]
    except:
        return ""

# Function to get AI response
def get_chat_response(text):
    """Generate a response using Google Gemini Flash, incorporating personal information."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"You are a chat bot that resembles me so answer like a human based on the following personal information, answer the query accurately and very precisely:\n\n{personal_info}\n\nUser: {text}\nBot:"
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "I'm sorry, I couldn't understand that."
    except Exception as e:
        print(f"Error in chatbot response: {e}")
        return "I'm sorry, I couldn't understand that."

# GUI Implementation
class VoiceChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Chatbot")
        
        self.text_display = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
        self.text_display.pack(pady=10)
        
        self.start_button = tk.Button(root, text="Start Chat", command=self.start_chat, font=("Arial", 14))
        self.start_button.pack(pady=5)
        
        self.stop_button = tk.Button(root, text="Stop Chat", command=self.stop_chat, font=("Arial", 14))
        self.stop_button.pack(pady=5)
        
        self.running = False
        self.text_display.insert(tk.END, "🤖 Bot: Hello! How may I assist you today?\n")
    
    def start_chat(self):
        if not self.running:
            self.running = True
            self.chat_loop()
    
    def stop_chat(self):
        self.running = False
        self.text_display.insert(tk.END, "🤖 Bot: Goodbye! Have a great day.\n")
        speak_response("Goodbye! Have a great day.")
    
    def chat_loop(self):
        if not self.running:
            return
        
        audio_file = record_audio(duration=5)
        if audio_file is None:
            self.text_display.insert(tk.END, "Error recording audio.\n")
            return
        
        user_text = transcribe_audio(audio_file)
        if not user_text.strip():
            self.text_display.insert(tk.END, "No input detected. Listening again...\n")
            self.root.after(1000, self.chat_loop)
            return
        
        self.text_display.insert(tk.END, f"🗣 User: {user_text}\n")
        if user_text.lower() in ["exit", "quit", "stop"]:
            self.stop_chat()
            return
        
        ai_response = get_chat_response(user_text)
        self.text_display.insert(tk.END, f"🤖 Bot: {ai_response}\n")
        speak_response(ai_response)
        
        self.root.after(1000, self.chat_loop)

# Start GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChatbotGUI(root)
    root.mainloop()
