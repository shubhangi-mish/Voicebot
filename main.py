import os
import torch
import sounddevice as sd
import numpy as np
import torchaudio
import pyttsx3
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not HF_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Missing API tokens. Please set HF_TOKEN and GEMINI_API_KEY in your .env file.")

# Set up device for inference
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper model and processor using Hugging Face token
model_id = "openai/whisper-small.en"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True, token=HF_TOKEN
).to(device)

processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)

# Initialize Whisper ASR pipeline
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

# Initialize pyttsx3 TTS
engine = pyttsx3.init()

def load_personal_info(filename="about_me.txt"):
    """Load personal information from a file."""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error loading personal information: {e}")
        return ""

personal_info = load_personal_info()

def speak_response(text):
    """Convert text to speech using pyttsx3."""
    print(f"🤖 Bot: {text}")
    engine.say(text)
    engine.runAndWait()

def record_audio(duration=5, samplerate=16000, filename="input.wav"):
    """Record user audio and save it as a WAV file."""
    try:
        print("🎤 Listening...")
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        write(filename, samplerate, audio)
        return filename
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

def transcribe_audio(filename):
    """Convert speech to text using Whisper ASR pipeline."""
    try:
        waveform, sample_rate = torchaudio.load(filename)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convert to numpy array for pipeline processing
        audio_data = {"array": waveform.squeeze(0).numpy(), "sampling_rate": 16000}

        result = asr_pipeline(audio_data)
        return result["text"]

    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def get_chat_response(text):
    """Generate a response using Google Gemini Flash, incorporating personal information."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"You are a chat bot that resembles me so answer like a human based on the following personal information, answer the query accurately:\n\n{personal_info}\n\nUser: {text}\nBot:"
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "I'm sorry, I couldn't understand that."
    except Exception as e:
        print(f"Error in chatbot response: {e}")
        return "I'm sorry, I couldn't understand that."

def voice_bot():
    """Main voice bot function to handle user interaction."""
    speak_response("Hello! How may I assist you today?")

    while True:
        audio_file = record_audio(duration=5)
        if audio_file is None:
            continue  # Skip if recording failed

        user_text = transcribe_audio(audio_file)

        if not user_text.strip():
            print("No input detected. Listening again...")
            continue

        print(f"🗣 User: {user_text}")

        if user_text.lower() in ["exit", "quit", "stop"]:
            speak_response("Goodbye! Have a great day.")
            break

        ai_response = get_chat_response(user_text)
        speak_response(ai_response)

if __name__ == "__main__":
    voice_bot()
