import streamlit as st
import numpy as np
import tensorflow as tf
import torch
import openai
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os

# --- 1. Load the pre-trained models and tokenizer ---

@st.cache_resource
def load_models():
    # Load your trained Keras model
    keras_model_path = "/run/media/theodoros/E/projects/hackathone/prompt_injection_model_lstm.keras"
    try:
        model_keras = tf.keras.models.load_model(keras_model_path)
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None, None, None, None
    
    # Load the multilingual-e5-base for embeddings
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    model_e5 = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_e5.to(device)
    
    return model_keras, tokenizer, model_e5, device

model_keras, tokenizer, model_e5, device = load_models()

# --- 2. Create the embedding function ---

def get_embeddings(texts):
    texts_with_prefix = [f"query: {text}" for text in texts]
    batch_dict = tokenizer(
        texts_with_prefix, 
        max_length=512, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    
    with torch.no_grad():
        outputs = model_e5(**batch_dict)
    
    embeddings = F.normalize(outputs.last_hidden_state.mean(dim=1), p=2, dim=1)
    
    return embeddings.cpu().numpy()

# --- 3. Streamlit UI and Logic ---

st.title("🛡️ Prompt Injection Guardrail")

# Hardcode the system prompt
SYSTEM_PROMPT = "You are a helpful AI assistant."

# Input for the User Prompt only
st.header("Input Prompt")
user_prompt = st.text_area("User Prompt")
    
if st.button("Submit"):
    if not user_prompt:
        st.warning("Please enter a user prompt.")
    else:
        # Combine hardcoded system prompt and user prompt to generate embeddings
        combined_prompt = SYSTEM_PROMPT + " " + user_prompt
        embeddings = get_embeddings([combined_prompt])
        
        # Make a prediction with your Keras model
        prediction = model_keras.predict(embeddings)[0][0]
        
        # Use the optimal threshold you found (replace 0.98 if needed)
        is_injection = (prediction > 0.98) 
        
        st.subheader("Model Guardrail Output")
        
        if is_injection:
            st.error("🚨 **Prompt Blocked!** Your prompt was classified as a potential injection.")
        else:
            st.success("✅ **Prompt Allowed!** Sending to LLM API.")
            
            # --- 4. Call LLM API (OpenAI) ---
            st.subheader("API Response (With Guardrail)")
            try:
                client = openai.OpenAI()
                with st.spinner("Generating LLM response..."):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")
                
        # --- 5. Show API response without the guardrail ---
        st.markdown("---")
        st.header("Raw API Response (Without Guardrail)")
        try:
            client = openai.OpenAI()
            with st.spinner("Generating raw LLM response..."):
                raw_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                st.write(raw_response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")