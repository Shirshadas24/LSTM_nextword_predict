import streamlit as st
import torch
import torch.nn as nn
import pickle
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# -----------------------------
# Rebuild model definition
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

# -----------------------------
# Helper: Convert tokens to indices
# -----------------------------
def text_to_indices(tokens, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

# -----------------------------
# Prediction function
# -----------------------------
def predict_next_word(model, vocab, input_text, max_len=61):
    tokens = word_tokenize(input_text.lower())
    indices = text_to_indices(tokens, vocab)
    padded = [0] * (max_len - len(indices)) + indices
    input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, dim=1)
    
    idx2word = {i: w for w, i in vocab.items()}
    return idx2word.get(predicted_idx.item(), '<unk>')

# -----------------------------
# Load model and vocab
# -----------------------------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
model = LSTMModel(vocab_size)
model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🔮 Next Word Prediction using LSTM")

user_input = st.text_input("Enter a phrase", placeholder="It is a beautiful")

if st.button("Predict Next Word"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        predicted_word = predict_next_word(model, vocab, user_input)
        st.success(f"🔤 Predicted next word: **{predicted_word}**")
