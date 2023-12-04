import tkinter as tk
from tkinter import scrolledtext, simpledialog
import threading
import requests
import json
import os
import subprocess
import pyaudio
import wave
import whisper

def text_to_speech(assistant_message):
    print("Starting text-to-speech...")  # Debugging print statement
    wav_path = "response.wav"  # Change this to the correct path if necessary
    command = f'echo "{assistant_message}" | .\\piper.exe -m .\\en_GB-alba-medium.onnx -f response.wav'
    subprocess.run(command, shell=True)

    try:
        audio = pyaudio.PyAudio()
        wf = wave.open(wav_path, 'rb')
        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Finished text-to-speech playback.")  # Debugging print statement
    except Exception as e:
        print(f"Error in text-to-speech playback: {e}")  # Error handling


# Define the server URL and headers
url = "http://10.0.0.17:5000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
character_name = "Example"  # Replace with dynamic input if needed
chat_history_file = character_name + '_chat_history.json'

# Load chat history from the file
def load_chat_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    return []

# Save chat history to the file
def save_chat_history(file_path, history):
    with open(file_path, 'w') as json_file:
        json.dump(history, json_file, indent=4)

history = load_chat_history(chat_history_file)

is_recording = False  # Global variable to track recording state

def start_stop_recording():
    global is_recording
    if not is_recording:
        # Start recording
        is_recording = True
        threading.Thread(target=record_audio, args=("input.wav",)).start()
        voice_button.config(text="Stop Recording")
    else:
        # Stop recording
        is_recording = False
        voice_button.config(text="Start Voice Input")

def record_audio(filename, duration=5):
    """Records audio for a given duration."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []

    for _ in range(0, int(44100 / 1024 * duration)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))

    transcribed_text = transcribe_audio(filename)
    
    # Enable the button and clear the recording message after recording
    voice_button.config(state=tk.NORMAL)

    send_message(transcribed_text)

def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

def handle_voice_input():
    """Handles voice input when the voice button is pressed."""
    update_chat_history("Recording . . .","")

    # Disable the button while recording
    voice_button.config(state=tk.DISABLED)
    
    # Start recording in a separate thread
    threading.Thread(target=record_audio, args=("input.wav", 5)).start()


def send_voice_message():
    audio_filename = "recorded_audio.wav"
    record_audio(audio_filename, duration=5)
    user_message = transcribe_audio(audio_filename)
    send_message(user_message)

def send_message(user_message=None, event=None):
    if not user_message:
        user_message = message_input.get("1.0", tk.END).strip()
        message_input.delete("1.0", tk.END)

    if user_message:
        history.append({"role": "user", "content": user_message})

        # Send data to the server
        data = {"mode": "chat", "character": character_name, "messages": history}
        response = requests.post(url, headers=headers, json=data, verify=False)
        assistant_message = response.json()['choices'][0]['message']['content']

        history.append({"role": "assistant", "content": assistant_message})
        update_chat_history(user_message, assistant_message)
        save_chat_history(chat_history_file, history)  # Save after each exchange
        threading.Thread(target=text_to_speech, args=(assistant_message,)).start()

# Function to update chat history in the GUI
def update_chat_history(user_message, assistant_message):
    chat_history.config(state='normal')
    chat_history.insert(tk.END, "You: " + user_message + "\n")
    chat_history.insert(tk.END, character_name + ": " + assistant_message + "\n")
    chat_history.yview(tk.END)
    chat_history.config(state='disabled')

# GUI Setup
root = tk.Tk()
root.withdraw()  # Hide the main window temporarily

# Prompt for bot's name
character_name = simpledialog.askstring("Bot Name", "Enter the name of the bot:", parent=root)
if not character_name:
    character_name = "DefaultBot"  # Default name if nothing is entered

chat_history_file = character_name + '_chat_history.json'
history = load_chat_history(chat_history_file)

root.deiconify()  # Show the main window again
root.title("Chat Application with " + character_name)

# Function to update chat history modified to use bot_name
def update_chat_history(user_message, assistant_message):
    chat_history.config(state='normal')
    chat_history.insert(tk.END, "You: " + user_message + "\n")
    chat_history.insert(tk.END, character_name + ": " + assistant_message + "\n")
    chat_history.yview(tk.END)
    chat_history.config(state='disabled')

frame = tk.Frame(root)
frame.grid(row=0, column=0, sticky="nsew")

# Configure the grid layout manager
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

# Chat history text box
chat_history = scrolledtext.ScrolledText(frame, state='disabled', height=15, width=50)
chat_history.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

# Message input text box
message_input = tk.Text(frame, height=3, width=40)
message_input.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

# Send button
send_button = tk.Button(frame, text="Send", command=send_message)
send_button.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

# Voice input button (if you have one)
voice_button = tk.Button(frame, text="Voice Input", command=lambda: threading.Thread(target=handle_voice_input).start())
voice_button.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

# Set focus and bring window to front
message_input.focus_set()
root.lift()
root.attributes('-topmost', True)
root.after_idle(root.attributes, '-topmost', False)

# Bind the Enter key to the send_message function in the message_input Text widget
message_input.bind("<Return>", send_message)

# To prevent the default behavior of the Enter key (which adds a new line)
# we need to stop the event propagation
def on_enter_pressed(event):
    send_message()
    return 'break'  # This stops the event from propagating further

message_input.bind("<Return>", on_enter_pressed)

root.mainloop()


