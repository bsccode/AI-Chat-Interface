import os
import json
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import whisper
import io
import requests
import subprocess
import threading
import random
import shlex
import simpleaudio as sa
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
from TTS.api import TTS

# Abstract base classes
class AIBase:
    def generate_response(self, message):
        raise NotImplementedError

class TTSBase:
    def speak(self, message):
        raise NotImplementedError

# OpenAI implementation
class OpenAIImpl(AIBase):
    def __init__(self, config):
        self.client = OpenAI(api_key=config["api_key"])
        self.config = config

    def generate_response(self, message):
        response = self.client.completions.create(
            model=self.config["model"],
            prompt=self.config["prompt"] + "\n\n" + message + "\n",
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            top_p=self.config["top_p"],
            frequency_penalty=self.config["frequency_penalty"],
            presence_penalty=self.config["presence_penalty"]
        )
        return response.choices[0].text

# ElevenLabs TTS implementation
class ELTTS(TTSBase):
    def __init__(self, config):
        self.config = config

    def speak(self, message):
        url = f'https://api.elevenlabs.io/v1/text-to-speech/{self.config["voice"]}'
        headers = {
            'accept': 'audio/mpeg',
            'xi-api-key': self.config["api_key"],
            'Content-Type': 'application/json'
        }
        data = {
            'text': message,
            'voice_settings': {
                'stability': 0.75,
                'similarity_boost': 0.75
            }
        }
        response = requests.post(url, headers=headers, json=data, stream=True)
        audio_content = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
        play(audio_content)

# Other AI and TTS implementations can be added here following the same pattern

class OobaboogaAI(AIBase):
    def __init__(self, config):
        self.url = config["endpoint"]
        self.mode = config.get("mode", "chat")  # Default to "chat" if not specified in config
        self.character = config.get("character", "Example")  # Default character if not specified

    def generate_response(self, message):
        history = [{"role": "user", "content": message}]
        data = {
            "mode": self.mode,
            "character": self.character,
            "messages": history
        }
        response = requests.post(self.url, headers={"Content-Type": "application/json"}, json=data, verify=False)
        assistant_message = response.json()['choices'][0]['message']['content']
        history.append({"role": "assistant", "content": assistant_message})
        return assistant_message
    
class Ollama(AIBase):
    def __init__(self, config):
        self.url = config["endpoint"]
        self.mode = config.get("mode", "chat")
        self.character = config.get("character", "Example")
        self.prompt = config["prompt"]


    def generate_response(self, message, history):

        # Append the new user message to the existing chat history
        history.append({"role": "user", "content": message})
        
        # Prepare the data to be sent to the server
        data = {
            "mode": self.mode,
            "prompt": self.prompt,
            "messages": history
        }

        # Send the request to the server and get the response
        response = requests.post(self.url, headers={"Content-Type": "application/json"}, json=data, verify=False)
        assistant_message = response.json()['choices'][0]['message']['content']

        return assistant_message
class SileroTTS(TTSBase):
    def __init__(self, config):
        self.endpoint = config["endpoint"]

    def speak(self, message):
        data = {'message': message}
        response = requests.post(self.endpoint, json=data)
        audio_content = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
        play(audio_content)  # Adjust if the response format is different

class CoquiTTS(TTSBase):
    def __init__(self, config):
        self.model_name = config["model_name"]
        self.output_path = config["output_path"]

    def speak(self, message):
        # Generate and save the audio file
        tts = TTS(model_name=self.model_name)
        tts.tts_to_file(text=message, file_path=self.output_path)

        # Play the audio file
        wave_obj = sa.WaveObject.from_wave_file(self.output_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait until the audio file is done playing

class PiperTTS(TTSBase):
    def __init__(self, config):
        self.model = config["model"]  # This should be just the filename, e.g., "en_GB-alba-medium.onnx"
        self.output_path = config["output_path"]

    def speak(self, message):
        # Escape single quotes and newlines in the message
        safe_message = message.replace("'", "'\\''").replace("\n", " ")

        # Construct the command to run Piper
        piper_executable = "./piper/piper"
        model_path = f"./piper/{self.model}"
        output_file_path = self.output_path

        command = f"echo '{safe_message}' | {piper_executable} --model {model_path} --output_file {output_file_path}"
        
        # Print the command for debugging
        print("Running command:", command)

        # Execute the command
        try:
            subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("Error running subprocess:", e)
            print("Standard Error Output:", e.stderr.decode())

        # Play the audio file
        wave_obj = sa.WaveObject.from_wave_file(output_file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        
class ChatHistoryManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_chat_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                return json.load(file)
        return []

    def save_chat_history(self, history):
        with open(self.file_path, 'w') as file:
            json.dump(history, file, indent=4)


class RunGUI:
    def __init__(self):

        self.history_manager = ChatHistoryManager("chat_history.json")  # Initialize ChatHistoryManager
        self.chat_history = self.history_manager.load_chat_history()  # Load chat history

        self.root = tk.Tk()
        self.root.title("Chat Interface")
        self.root.geometry("600x600")



        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.record_button = ttk.Button(frame, text="Start Recording", command=self.start_recording_thread)
        self.record_button.pack(pady=5)

        self.submit_text_button = ttk.Button(frame, text="Submit Text", command=self.start_submit_text_thread)
        self.submit_text_button.pack(pady=5)

        # Input Text Box for Manual Text Entry
        self.input_text = tk.Text(frame, height=4)
        self.input_text.pack(pady=5, fill=tk.X)
        
        # Output/History Text Box
        self.output_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.output_text.pack(pady=5, fill=tk.BOTH, expand=True)

        # Load configuration
        self.config = self.load_configuration("config.json")
        self.ai_service = create_ai_service(self.config)
        self.tts_service = create_tts_service(self.config)

        # AI API Selector
        self.ai_api_label = ttk.Label(frame, text="Select AI API:")
        self.ai_api_label.pack(pady=5)
        self.ai_api_selector = ttk.Combobox(frame, values=["OpenAI", "Oobabooga","Ollama"], state="readonly")
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

    def load_configuration(self, file_path):
        with open(file_path, "r") as json_file:
            return json.load(json_file)

    def start_recording_thread(self):
        threading.Thread(target=self.record_and_transcribe, daemon=True).start()
    
    def start_submit_text_thread(self):
        threading.Thread(target=self.submit_text, daemon=True).start()
    
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
        audio = record_audio(duration=5)
        audio_filename = 'recorded_audio.wav'
        save_audio_as_wav(audio, audio_filename)
        transcribed_text = transcribe_with_whisper(audio_filename)
        self.update_transcription_text(transcribed_text)
        save_transcription_to_file(transcribed_text, 'transcription.txt')
        self.backend()  # Call backend after saving the transcription
        self.record_button["state"] = "normal"

    def backend(self):
        # Load the configuration
        config = self.load_configuration("config.json")

        # Create AI and TTS service instances
        ai_service = create_ai_service(config)
        tts_service = create_tts_service(config)

        # Load the user message from the file or GUI
        user_message = self.load_user_message("transcription.txt")

        # Generate the AI response (ensure that generate_response method accepts chat history)
        assistant_message = ai_service.generate_response(user_message, self.chat_history)

        # Update the GUI with the AI response (if necessary)
        self.update_transcription_text(assistant_message)

        # Update the chat history
        self.chat_history.append({"role": "assistant", "content": assistant_message})

        # Save the updated chat history
        self.history_manager.save_chat_history(self.chat_history)

        # Use TTS service to speak out the assistant's response
        tts_service.speak(assistant_message)

    def load_user_message(self, file_path):
        # Implementation to load the user's message from the file or GUI
        # Example:
        with open(file_path, 'r') as file:
            return file.read().strip()


    def submit_text(self):
        entered_text = self.input_text.get('1.0', tk.END).strip()
        # Now, 'entered_text' contains the text from the scrolled text box
        save_transcription_to_file(entered_text, 'transcription.txt')
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, "User: " + entered_text + "\n")
        self.input_text.delete('1.0', tk.END)
        self.backend()  # Call backend after saving the manual text

    def update_transcription_text(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END,"AI: " + text + "\n")
        self.output_text.config(state=tk.DISABLED)

    def process_text_file(self, file_path, ai_service, tts_service):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()
            if line:
                response = ai_service.generate_response(line)
                self.output_text.config(state=tk.NORMAL)
                self.output_text.insert(tk.END, "AI: " + response + "\n")
        self.output_text.config(state=tk.DISABLED)
        tts_service.speak(response)

    def run(self):
        self.root.mainloop()

# Factory functions to create service instances
def create_ai_service(config):
    service_name = config["active_services"]["AI"]
    if service_name == "OpenAI":
        return OpenAIImpl(config["services"]["AI"]["OpenAI"])
    elif service_name == "Oobabooga":
        return OobaboogaAI(config["services"]["AI"]["Oobabooga"])
    elif service_name == "Ollama":
        return Ollama(config["services"]["AI"]["Ollama"])
    # Add additional elif conditions for other AI services as needed

def create_tts_service(config):
    service_name = config["active_services"]["TTS"]
    if service_name == "ElevenLabs":
        return ELTTS(config["services"]["TTS"]["ElevenLabs"])
    elif service_name == "Silero":
        return SileroTTS(config["services"]["TTS"]["Silero"])
    elif service_name == "CoquiTTS":
        return CoquiTTS(config["services"]["TTS"]["CoquiTTS"])
    elif service_name == "piper":
        return PiperTTS(config["services"]["TTS"]["piper"])
    # Add additional elif conditions for other TTS services as needed

# Load configuration from a JSON file
def load_configuration():
    with open("config.json", "r") as json_file:
        return json.load(json_file)

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
    
if __name__ == "__main__":
    gui = RunGUI()
    gui.root.mainloop()