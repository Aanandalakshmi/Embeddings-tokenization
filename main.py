import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
    model=AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer,model=load_model()

def get_embeddings(text):
    input=tokenizer(text,return_tensors="pt",truncation=True,max_length=512)
    with torch.no_grad():
        output=model(**input)
        cls_embedding=output.last_hidden_state[:,0,:]
        return cls_embedding

st.title("Tokenization & Embediing Visualizer")

text1=st.text_input("Enter first word or sentence")
text2=st.text_input("Enter second word or sentence")

if st.button("Compare Embeddings"):
    emb1=get_embeddings(text1)
    emb2=get_embeddings(text2)

    sim_score= cosine_similarity(emb1.numpy(),emb2.numpy())[0][0]

    st.markdown("###Tokeniztion")
    st.write(f"**{text1}** -> {tokenizer.tokenize(text1)} ")
    st.write(f"**{text2}** -> {tokenizer.tokenize(text2)}")

    st.markdown("### Cosine similarity")
    st.success(f"*Semantic Similarity score:* {sim_score:.4f}")