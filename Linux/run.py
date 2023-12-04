import json
import io
import time
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

# Factory functions to create service instances
def create_ai_service(config):
    service_name = config["active_services"]["AI"]
    if service_name == "OpenAI":
        return OpenAIImpl(config["services"]["AI"]["OpenAI"])
    elif service_name == "Oobabooga":
        return OobaboogaAI(config["services"]["AI"]["Oobabooga"])
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

# Process the text file with AI and TTS
def process_text_file(file_path, ai_service, tts_service):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line:
            print(f"Processing line: {line}")
            response = ai_service.generate_response(line)
            print(response)
            tts_service.speak(response)

# Main script execution
if __name__ == "__main__":
    config = load_configuration()
    ai_service = create_ai_service(config)
    tts_service = create_tts_service(config)

    file_path = "transcription.txt"
    print("\nRunning!\n\n")
    process_text_file(file_path, ai_service, tts_service)
    print("\nFinished processing the file.\n")