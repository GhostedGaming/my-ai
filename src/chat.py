import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('goofy_ai_model.h5')

# Load the tokenizer from pickle file
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_sequence_len = 20  # This should match what you used during training

def sample_with_temperature(preds, temperature=0.7):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature  # Avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_response(user_message, num_words=12, temperature=0.7):
    user_message = user_message.lower().strip()
    tokenized_input = tokenizer.texts_to_sequences([user_message])[0] if user_message else []

    generated_tokens = tokenized_input.copy()

    for _ in range(num_words):
        padded_sequence = pad_sequences([generated_tokens], maxlen=max_sequence_len - 1, padding='pre')
        preds = model.predict(padded_sequence, verbose=0)[0]

        next_index = sample_with_temperature(preds, temperature)
        # Convert predicted index back to word
        output_word = None
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                output_word = word
                break

        if output_word is None:
            break  # No valid word predicted, stop generation

        generated_tokens.append(next_index)

    # Decode tokens back to words
    response_words = []
    for token in generated_tokens:
        for word, index in tokenizer.word_index.items():
            if index == token:
                response_words.append(word)
                break

    return ' '.join(response_words).strip()

def main():
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == 'exit':
            print("Goodbye!")
            break
        response = generate_response(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()