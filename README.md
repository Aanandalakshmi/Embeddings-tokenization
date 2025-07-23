
## 🧠 Tokenization & Embedding Visualizer with BERT

This interactive Streamlit app lets you explore how **transformer-based language models** like **BERT** tokenize text and generate contextual **word/sentence embeddings**. You can input two sentences or words and visualize their tokenization along with a **semantic similarity score** based on **cosine similarity** of their `[CLS]` embeddings.

### 🚀 Features

* 🔤 Tokenize input text using `bert-base-uncased` tokenizer
* 🧬 Generate contextual embeddings using the BERT model
* 🧠 Compare semantic similarity between two inputs
* 📈 Visual display of tokenization and similarity score
* 💡 Useful for understanding how embeddings and tokenization work in transformer models

### 🧰 Technologies Used

* [🤗 Transformers (BERT)](https://huggingface.co/bert-base-uncased) – for tokenization and embedding generation
* [Streamlit](https://streamlit.io) – for building the interactive web app
* [PyTorch](https://pytorch.org) – underlying framework for running the model
* [Scikit-learn](https://scikit-learn.org) – for computing cosine similarity



I built this project to get a clearer understanding of how **embeddings**, **tokenization**, and **semantic similarity** work under the hood using pre-trained language models. It’s a hands-on way to visualize what happens when you input text into BERT and how the model represents meaning.

