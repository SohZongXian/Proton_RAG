# Proton_RAG

Proton_RAG is a chatbot that provides information regarding a Malaysian car called Proton Saga. It uses RAG (Retrieval-Augmented Generation) to provide context from a pre-loaded PDF document that contains more information about the Proton Saga. The chatbot can answer questions about the car's features, specifications, history, and price. It can also generate creative content related to the car, such as poems, stories, code, and songs.

## Architecture

The chatbot is built using the following components:

- **Google Gemini Pro**: This is the language model (LLM) that generates the chatbot responses. It is a state-of-the-art multimodal model that can reason across text, images, video, audio, and code. It is available through the [Gemini API] from Google AI for Developers. The chatbot uses the `generate_text` method to produce natural language outputs based on the user inputs and the retrieved context.
- **VectorStore from LangChain**: This is the vector storage and retrieval system that enables fast and scalable similarity search. It is based on the [Faiss] library developed by Facebook AI Research. It allows the chatbot to index and query the PDF document using vector embeddings. The chatbot uses the `VectorStore` and `VectorStoreRetriever` classes from the [langchain_core.vectorstores] module to create and access the vector index.
- **GoogleGenerativeAIEmbeddings**: This is the embedding service in the Gemini API that generates state-of-the-art embeddings for words, phrases, and sentences. The resulting embeddings can then be used for NLP tasks, such as semantic search, text classification, and clustering, among many others. The chatbot uses the `generate_embeddings` method to create vector representations of the user inputs and the PDF document.
- **FAISS index**: This is the data structure that stores the vector embeddings and supports efficient nearest neighbor search. It is implemented by the Faiss library and accessed by the VectorStore from LangChain. The chatbot uses the `IndexIVFFlat` class to create an inverted file index with exact post-verification. This allows the chatbot to retrieve the most relevant passages from the PDF document based on the user inputs.

The chatbot architecture can be summarized by the following diagram:

```mermaid
graph LR
    A[User Input] -->|generate_embeddings| B[Query Embedding]
    B -->|VectorStoreRetriever| C[FAISS Index]
    C -->|VectorStoreRetriever| D[Retrieved Context]
    D -->|generate_text| E[Chatbot Response]
    E --> A
    F[PDF Document] -->|generate_embeddings| G[Document Embeddings]
    G -->|VectorStore| C

Usage
To use the chatbot, you need to have the following prerequisites:

Python 3.9 or higher
Conda package manager
Google AI Studio account

To install the chatbot, follow these steps:

Clone this repository to your local machine.
Create a conda environment with the required dependencies: conda env create -f environment.yml
Activate the conda environment: conda activate proton_rag
install dependencies using requirement.txt
Set the environment variables for the Gemini API: export GEMINI_API_KEY=your_gemini_api_key
Run the chatbot script: streamlit run guide.py
