import os
import json
import random
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sounddevice as sd
import numpy as np
from run import create_ai_service, create_tts_service
import scipy.io.wavfile as wavfile
import whisper

def record_audio(duration=5, fs=44100):
    """Record audio from the microphone."""
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return recording

def save_audio_as_wav(audio, filename, fs=44100):
    """Save the recorded audio as a WAV file."""
    audio = np.squeeze(audio)  # Remove channel dimension if mono
    wavfile.write(filename, fs, audio)

def transcribe_with_whisper(audio_file, model_name='base'):
    """Transcribe audio using Whisper."""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file)
    return result['text']

def save_transcription_to_file(text, filename):
    """Save the transcription to a text file."""
    with open(filename, 'w') as file:
        file.write(text)

def load_prompts(filename):
    with open(filename, 'r') as file:
        return json.load(file)

class RunGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio Transcription")
        self.root.geometry("500x500")

        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Load configuration
        self.config = self.load_configuration("config.json")

        self.ai_service = create_ai_service(self.config)
        self.tts_service = create_tts_service(self.config)

        # AI API Selector
        self.ai_api_label = ttk.Label(frame, text="Select AI API:")
        self.ai_api_label.pack(pady=5)
        self.ai_api_selector = ttk.Combobox(frame, values=["OpenAI", "Oobabooga"], state="readonly")
        self.ai_api_selector.set(self.config["active_services"]["AI"])
        self.ai_api_selector.pack(pady=5)
        self.ai_api_selector.bind("<<ComboboxSelected>>", self.update_ai_api)

        # TTS API Selector
        self.tts_api_label = ttk.Label(frame, text="Select TTS API:")
        self.tts_api_label.pack(pady=5)
        self.tts_api_selector = ttk.Combobox(frame, values=["ElevenLabs", "CoquiTTS", "piper"], state="readonly")
        self.tts_api_selector.set(self.config["active_services"]["TTS"])
        self.tts_api_selector.pack(pady=5)
        self.tts_api_selector.bind("<<ComboboxSelected>>", self.update_tts_api)

        self.record_button = ttk.Button(frame, text="Start Recording", command=self.start_recording_thread)
        self.record_button.pack(pady=5)

        self.manual_text_entry = tk.Entry(frame)
        self.manual_text_entry.pack(pady=5, fill=tk.X)

        self.submit_text_button = ttk.Button(frame, text="Submit Text", command=self.submit_text)
        self.submit_text_button.pack(pady=5)

        self.transcription_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        self.transcription_text.pack(pady=5, fill=tk.BOTH, expand=True)


    def load_configuration(self, file_path):
        with open(file_path, "r") as json_file:
            return json.load(json_file)

    def start_recording_thread(self):
        threading.Thread(target=self.record_and_transcribe, daemon=True).start()
    
    def update_ai_api(self, event=None):
        selected_ai_api = self.ai_api_selector.get()
        self.config["active_services"]["AI"] = selected_ai_api
        self.save_configuration("config.json")

    def update_tts_api(self, event=None):
        selected_tts_api = self.tts_api_selector.get()
        self.config["active_services"]["TTS"] = selected_tts_api
        self.save_configuration("config.json")

    def save_configuration(self, file_path):
        with open(file_path, "w") as json_file:
            json.dump(self.config, json_file, indent=4)
    
    def start_timer(self):
        # Call this method whenever you want to start or reset the timer
        self.stop_timer()
        self.timer = threading.Timer(self.idle_timeout, self.trigger_idle_response)
        self.timer.start()

    def stop_timer(self):
        if hasattr(self, 'timer') and self.timer.is_alive():
            self.timer.stop_timer()

    def trigger_idle_response(self):
        prompt = random.choice(self.prompts)
        print(f"Idle Triggered: {prompt}")
        response = self.ai_service.generate_response(prompt)
        threading.Thread(target=self.speak_response, args=(response,), daemon=True).start()

    def speak_response(self, response):
        self.tts_service.speak(response)


    def some_user_activity_method(self):
        # Call this method whenever there is user activity
        self.timer.reset_timer()

    def record_and_transcribe(self):
        self.record_button["state"] = "disabled"
        audio = record_audio(duration=10)  # 5 seconds recording
        audio_filename = 'recorded_audio.wav'
        save_audio_as_wav(audio, audio_filename)
        transcribed_text = transcribe_with_whisper(audio_filename)
        self.update_transcription_text(transcribed_text)
        save_transcription_to_file(transcribed_text, 'transcription.txt')
        os.system("python run.py")  # Call run.py after saving the transcription
        self.record_button["state"] = "normal"


    def submit_text(self):
        entered_text = self.transcription_text.get('1.0', tk.END).strip()
        # Now, 'entered_text' contains the text from the scrolled text box
        save_transcription_to_file(entered_text, 'transcription.txt')
        os.system("python run.py")  # Call run.py after saving the manual text


    def update_transcription_text(self, text):
        self.transcription_text.delete('1.0', tk.END)
        self.transcription_text.insert(tk.END, text)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = RunGUI()
    gui.run()