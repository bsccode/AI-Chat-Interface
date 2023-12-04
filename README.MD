# Voice-Interactive Chat Application

## Description
This project is a voice-interactive chat application built in Python. It utilizes OpenAI's Whisper model for audio transcription and PyAudio for audio recording and playback. The application allows users to interact with a chatbot either through text input or voice commands. The GUI is built using Tkinter, providing a user-friendly interface for interaction. The current back end is set up and designed for use with oobabooga's text-generation-webui and the text to speech is using Piper-TTS. It is currently set for Open-AI API format.

## Features
- Text and voice input modes.
- Speech-to-text transcription using OpenAI's Whisper model.
- Text-to-speech functionality for chatbot responses.
- Dynamic bot name assignment through a simple dialog prompt.
- Chat history display and storage.
- Customizable and expandable GUI layout.

## Installation
To run this application, you need to have Python installed on your system along with several dependencies. Below are the steps to set up and run the project:

### Prerequisites
- Python 3.11.6 (Python used to develop)
- Pip (Python package installer)

### Dependencies
Create Virtual environment:
    python -m venv venv

Activate environment:
    venv\Scripts\activate

Install the required dependencies by running the following command in your terminal:
    pip install -r requirements.txt

### Running the Application
To start the application, navigate to the project directory in your terminal and run:
    python tk.py

## Usage
Upon launching the application, you will be prompted to enter the name of the chatbot (Character name as in Oobabooga web ui). Afterward, the main chat window will open, allowing you to interact with the bot through text or voice input.

Chat histories are generated automatically and sent to API from (character_name)_chat_history.json.

config.json is not in use at this point but will be used in future to enable easy changing of API's used.

Piper-TTS is currently used for low latency and decent quality TTS.

You can communicate with bot through either microphone audio or through text. 

## Contributing
Contributions to this project are welcome.

## Links to Programs in use

https://github.com/oobabooga/text-generation-webui

https://github.com/rhasspy/piper


## License
[MIT](https://choosealicense.com/licenses/mit/)

