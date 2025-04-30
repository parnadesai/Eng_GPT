import streamlit as st
import torch
from bigram import BigramLanguageModel, encode, decode

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model once
@st.cache_resource
def load_model():
    model = BigramLanguageModel()
    model.load_state_dict(torch.load("trained_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("🧠 Character-Level Text Generator")
st.markdown("Generate text using a Transformer-based character-level language model trained on Shakespeare's works.")

# Prompt input
prompt = st.text_input("📝 Enter a prompt", value="ROMEO:")
max_new_tokens = st.slider("📏 Characters to generate", min_value=50, max_value=1000, value=200)

# Generate text
if st.button("🚀 Generate Text"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            with torch.no_grad():
                encoded = encode(prompt)
                if len(encoded) == 0:
                    st.error("Encoding failed: prompt may contain unknown characters.")
                else:
                    idx = torch.tensor([encoded], dtype=torch.long).to(device)
                    out = model.generate(idx, max_new_tokens=max_new_tokens)
                    result = decode(out[0].tolist())

                    st.markdown("### ✍️ Generated Text")
                    st.text_area("Result", result, height=400)
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
