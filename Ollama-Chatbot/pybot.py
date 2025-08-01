from openai import OpenAI
import json
import os

client = OpenAI(
    base_url='http://10.0.0.32:11434/v1',
    api_key='ollama',  # Required, but unused in this context
)

character_name = input("Who are you talking to today? ").lower()

if os.path.exists(f"characters\{character_name}_context.txt"):
    with open(f"characters\{character_name}_context.txt", 'r') as file:
        context = file
else:
    backstory = input("Please enter your characters backstory and context for this chat: ")
    with open(f"characters\{character_name}_context.txt", "w") as f:
      f.write(backstory)


# Load initial context from a file
with open(f"characters\{character_name}_context.txt", "r") as file:
    context = file.read()

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

# Initialize ChatHistoryManager
history_manager = ChatHistoryManager(f"characters\{character_name}_chat_history.json")
# Load chat history
chat_history = history_manager.load_chat_history()

#append context to remind chat bot
initial_context = chat_history.append({"role": "system", "content": context})

# Save the updated chat history
history_manager.save_chat_history(chat_history)

# Prepare the initial message list for the conversation
initial_messages = [{"role": "system", "content": context}] + chat_history


# Generate the initial response
initial_response = client.chat.completions.create(
  model="dolphin-mixtral",
  messages=initial_messages
)

# Display the context and the AI's initial response (if any)
print("\nContext: " + context)
if initial_response.choices:
    assistant_message = initial_response.choices[0].message.content
    #print("\nAI: " + assistant_message)

# Main chat loop
while True:
    user_input = input("\nYou: ")
    
    # Append user's message to the history
    chat_history.append({"role": "user", "content": user_input})

    # Prepare the message list, including the system, previous exchanges, and the new user input
    messages_for_completion = [{"role": "system", "content": context}] + chat_history

    # Generate response based on the updated history
    response = client.chat.completions.create(
        model="dolphin-mixtral",
        messages=messages_for_completion
    )
    
    if response.choices:
        assistant_message = response.choices[0].message.content
        print("\nAI: " + assistant_message)
        
        # Append assistant's message to the history
        chat_history.append({"role": "assistant", "content": assistant_message})
        
        # Save the updated chat history
        history_manager.save_chat_history(chat_history)
