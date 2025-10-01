import numpy as np
import pickle
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("="*60)
print("Loading Goofy AI Chatbot...")
print("="*60)

try:
    model = load_model('goofy_ai_model.h5')
    print("Model loaded successfully")
except:
    print("Error: 'goofy_ai_model.h5' not found")
    print("Run the training script first to create the model")
    exit()

try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully")
except:
    print("Error: 'tokenizer.pkl' not found")
    print("Run the training script first to create the tokenizer")
    exit()

max_sequence_len = model.input_shape[1] + 1
print(f"Max sequence length: {max_sequence_len}")
print("="*60)

response_templates = [
    "i think",
    "yeah i",
    "that's so",
    "honestly i",
    "oh i",
    "well i",
    "i feel",
    "i love",
    "that sounds",
    "wow that's",
    "i understand",
    "maybe we",
    "definitely",
    "probably",
    "no way",
    "for real",
    "i guess",
    "actually",
]

def generate_response(user_message, num_words=12, temperature=0.7):
    seed_text = random.choice(response_templates)
    
    generated = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated])[0]
        
        if not token_list:
            words = generated.split()
            if words:
                token_list = tokenizer.texts_to_sequences([words[-1]])[0]
        
        if not token_list:
            break
        
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predictions = model.predict(token_list, verbose=0)[0]
        
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        predicted = np.random.choice(len(predictions), p=predictions)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        if output_word:
            generated += " " + output_word
        else:
            break
    
    return generated

def chat():
    print("="*60)
    print("Commands:")
    print("  Type anything to chat")
    print("  'temp X' to change temperature (0.5=boring, 1.2=crazy)")
    print("  'words X' to change response length")
    print("  'quit' to exit")
    print("="*60)
    
    temperature = 0.7
    num_words = 12
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                temperature = max(0.1, min(2.0, temperature))
                print(f"Temperature set to {temperature}")
                continue
            except:
                print("Invalid. Use: temp 0.8")
                continue
        
        if user_input.lower().startswith('words '):
            try:
                num_words = int(user_input.split()[1])
                num_words = max(5, min(25, num_words))
                print(f"Length set to {num_words} words")
                continue
            except:
                print("Invalid. Use: words 12")
                continue
        
        response = generate_response(user_input, num_words=num_words, temperature=temperature)
        print(f"AI: {response}")

if __name__ == "__main__":
    chat()