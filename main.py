import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from nltk_utils import tokenize, bag_of_words  # Ensure nltk_utils.py is present

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChatbotAssistant:
    def __init__(self, intents_path, dimensions_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.function_mappings = function_mappings
        self.memory = {}

        # Load dimensions and vocabulary
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.vocabulary = dimensions["all_words"]
        self.intents = dimensions["tags"]
        self.intents_responses = {}  # Added to store responses

        # Load intents data to fill responses
        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)
            for intent in intents_data["intents"]:
                self.intents_responses[intent["tag"]] = intent["responses"]

        self.model = ChatbotModel(dimensions["input_size"], dimensions["output_size"])

    def load_model(self, model_path):
        """Loads the trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError("Error: Model file not found!")

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_message(self, input_message):
        """Processes user input and returns a chatbot response."""
        words = tokenize(input_message)
        print(f"Tokenized words: {words}")

        bag = bag_of_words(words, self.vocabulary)
        print(f"Bag of words length: {len(bag)}, Expected input size: {self.model.fc1.in_features}")

        bag_tensor = torch.tensor(np.array(bag, dtype=np.float32).reshape(1, -1))
        print(f"Bag tensor shape: {bag_tensor.shape}")

        with torch.no_grad():
            predictions = self.model(bag_tensor)
            probabilities = F.softmax(predictions, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, dim=1)

        if confidence.item() < 0.6:
            return "I'm sorry, I didn't understand that."

        predicted_intent = self.intents[predicted_class_index.item()]
        print(f"Predicted intent: {predicted_intent}")  # Debugging

        # Check if intent is mapped to a function
        if self.function_mappings and predicted_intent in self.function_mappings:
            return self.function_mappings[predicted_intent]()

        # Return a random response from intents.json
        return random.choice(self.intents_responses.get(predicted_intent, ["I'm sorry, I don't understand."]))


def get_joke():
    """Returns a random programming joke."""
    jokes = [
        "Why do programmers hate nature? It has too many bugs!",
        "Why do Java developers wear glasses? Because they don’t see sharp!",
        "How many programmers does it take to change a light bulb? None, it's a hardware problem!"
    ]
    return random.choice(jokes)


if __name__ == '__main__':
    try:
        assistant = ChatbotAssistant('intents.json', 'dimensions.json', function_mappings={'joke': get_joke})
        assistant.load_model('chatbot_model.pth')

        while True:
            message = input("You: ")
            if message.lower() == '/quit':
                print("Bot: Goodbye!")
                break
            print(f"Bot: {assistant.process_message(message)}")

    except FileNotFoundError as e:
        print(e)
