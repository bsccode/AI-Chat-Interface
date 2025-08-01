# Import necessary libraries
from openai import OpenAI
import json
import os

# Function to initialize the OpenAI client
def initialize_openai_client():
    return OpenAI(
        base_url='http://10.0.0.32:11434/v1',
        api_key='ollama',
    )

# Class for managing character context
class CharacterContextManager:
    def __init__(self, character_name):
        self.character_name = character_name
        self.file_path = f"characters\\{character_name}_context.txt"

    def load_or_create_context(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                return file.read()
        else:
            backstory = input("Please enter your character's backstory and context for this chat: ")
            with open(self.file_path, "w") as f:
                f.write(backstory)
            return backstory

# Class for managing chat history
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

def main():
    client = initialize_openai_client()
    character_name = input("Who are you talking to today? ").lower()
    context_manager = CharacterContextManager(character_name)
    context = context_manager.load_or_create_context()

    history_manager = ChatHistoryManager(f"characters\\{character_name}_chat_history.json")
    chat_history = history_manager.load_chat_history()

    if not chat_history:  # If there's no history, we add the initial context
        chat_history.append({"role": "system", "content": context})
        history_manager.save_chat_history(chat_history)

    print("\nContext: " + context)

    # Generate and display initial AI response based on the context
    initial_messages = [{"role": "system", "content": context}]
    initial_response = client.chat.completions.create(
        model="wizard-vicuna-uncensored",
        messages=initial_messages
    )

    if initial_response.choices:
        initial_ai_message = initial_response.choices[0].message.content
        print("\nAI: " + initial_ai_message)
        chat_history.append({"role": "assistant", "content": initial_ai_message})
        history_manager.save_chat_history(chat_history)

    # Main chat loop
    while True:
        user_input = input("\nYou: ")
        chat_history.append({"role": "user", "content": user_input})

        messages_for_completion = [{"role": "system", "content": context}] + chat_history

        response = client.chat.completions.create(
            model="mistral",
            messages=messages_for_completion
        )

        if response.choices:
            assistant_message = response.choices[0].message.content
            print("\nAI: " + assistant_message)

            chat_history.append({"role": "assistant", "content": assistant_message})
            history_manager.save_chat_history(chat_history)

if __name__ == "__main__":
    main()
