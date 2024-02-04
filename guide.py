from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
import streamlit as st

genai.configure(api_key=st.secrets["api_key"])

# Chat model initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", google_api_key=st.secrets["api_key"]
)

# Prompt template
prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details in a nicely formatted point form. If the answer is not in
    the provided context, provide additional information from your knowledge base.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=prompt_template
)


# Function to get PDF text
def get_pdf_text(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to get text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to get vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["api_key"],
    )
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


# # Load PDF and create vector store
# vectorstore = get_vector_store(
#     get_text_chunks(
#         get_pdf_text(
#             r"D:\Development\Python\Aggregate_website\Proton_Saga_3rd_Gen_Guide.pdf"
#         )
#     )
# )

# Load Q&A Chain
retrieval_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)


# Streamlit app
def main():
    st.title("Proton Saga Guide Malaysia")

    # User input
    user_input = st.text_input("Enter your question:")

    if user_input:
        with st.spinner("Generating Answer..."):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=st.secrets["api_key"],
            )
            new_db = FAISS.load_local("faiss_index", embeddings)
            docs = new_db.similarity_search(user_input)
            response = retrieval_chain(
                {"input_documents": docs, "question": user_input},
                return_only_outputs=True,
            )
            st.write(f"{response['output_text']}")


if __name__ == "__main__":
    main()
