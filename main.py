import pandas as pd
import numpy as np
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam

print("Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found, using CPU")

sentences = []

try:
    df_slang = pd.read_csv('all_slangs.csv')
    sentences.extend(df_slang['Example'].tolist())
    print(f"Loaded {len(df_slang)} slang examples")
except FileNotFoundError:
    print("all_slangs.csv not found")

try:
    df_english = pd.read_csv('english.csv')
    if 'def' in df_english.columns:
        definitions = df_english['def'].dropna().astype(str).tolist()
        definitions = [d for d in definitions if 3 <= len(d.split()) <= 20]
        
        import random
        random.shuffle(definitions)
        definitions = definitions[:40000]
        
        sentences.extend(definitions)
        print(f"Loaded {len(definitions)} dictionary definitions")
except FileNotFoundError:
    print("english.csv not found")

conversational = [
    "hello how are you doing today",
    "hi there nice to meet you",
    "hey what's up with you lately",
    "good morning have a great day",
    "good afternoon hope you're well",
    "good evening how was your day",
    "that's really cool and interesting",
    "i love that so much right now",
    "this is amazing and wonderful stuff",
    "wow that's incredible news today",
    "oh no that's terrible to hear",
    "haha that's so funny man",
    "i understand what you mean exactly",
    "that makes sense to me now",
    "i don't know about that honestly",
    "maybe we should try something else",
    "i think that's a good idea",
    "sounds good to me let's do it",
    "no way that's crazy talk",
    "for real that's wild stuff",
    "honestly i feel the same way",
    "yeah i totally agree with you",
    "nah i don't think so really",
    "absolutely that's completely right",
    "definitely the best option here",
    "probably not the best choice",
    "hopefully things get better soon",
    "i'm excited about this project",
    "i'm tired of waiting around",
    "i'm happy to help you out",
    "thanks for your help today",
    "you're welcome anytime friend",
    "no problem at all really",
    "sorry about that my mistake",
    "it's okay don't worry about it",
    "what do you think about that",
    "how do you feel about this",
    "when did that happen exactly",
    "where are you going today",
    "why did you do that thing",
    "who told you about this",
    "i can't believe this happened today",
    "that was the best experience ever",
    "this has been a long day",
    "i need to get some rest",
    "let's talk about something else now",
    "tell me more about your day",
    "what happened after that moment",
    "how did everything turn out",
    "that sounds really interesting to me",
    "i've never heard of that before",
] * 200

sentences.extend(conversational)

if not sentences:
    sentences = ["That's so cool!", "I love this.", "This is amazing!"] * 100

print(f"Total training sentences: {len(sentences)}")

print("Tokenizing...")
tokenizer = Tokenizer(num_words=15000)
tokenizer.fit_on_texts(sentences)
total_words = min(len(tokenizer.word_index) + 1, 15000)
print(f"Vocabulary size: {total_words}")

print("Creating training sequences...")
input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, min(len(token_list), 25)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = min(max([len(x) for x in input_sequences]), 25)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print(f"Training patterns: {len(X):,}")
print(f"Max sequence length: {max_sequence_len}")

model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    LSTM(200, return_sequences=True),
    Dropout(0.2),
    LSTM(200),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(learning_rate=0.002), 
              metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

print("\n" + "="*60)
print("TRAINING...")
print("="*60)

samples_per_sec = 1500
total_samples = len(X) * 30
estimated_seconds = total_samples / samples_per_sec
print(f"Estimated time: ~{int(estimated_seconds/60)} minutes")
print("="*60)

history = model.fit(X, y, 
                   epochs=30, 
                   batch_size=256, 
                   verbose=1, 
                   validation_split=0.1)

print("\nTRAINING COMPLETE!")

model.save('goofy_ai_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Model saved!")

def generate_text(seed_text, next_words=15, temperature=0.6):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        if not token_list:
            words = seed_text.split()
            if words:
                token_list = tokenizer.texts_to_sequences([words[-1]])[0]
        
        if not token_list:
            seed_text = "that is"
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        if not token_list:
            return seed_text
        
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
            seed_text += " " + output_word
        else:
            break
    
    return seed_text

def chat():
    print("\n" + "="*60)
    print("CHAT MODE")
    print("="*60)
    print("Commands: 'temp X' | 'words X' | 'quit'")
    print("="*60)
    
    temperature = 0.6
    num_words = 15
    
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
                print(f"Temperature: {temperature}")
                continue
            except:
                print("Invalid")
                continue
        
        if user_input.lower().startswith('words '):
            try:
                num_words = int(user_input.split()[1])
                print(f"Length: {num_words}")
                continue
            except:
                print("Invalid")
                continue
        
        response = generate_text(user_input, next_words=num_words, temperature=temperature)
        print(f"AI: {response}")

chat()