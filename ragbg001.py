import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader

def find_pdf_files(folder):
    pdf_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def load_documents_from_pdfs(pdf_paths):
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            # Optionally add file path as metadata
            doc.metadata["source"] = path
            documents.append(doc)
    return documents

if __name__ == "__main__":
    pdf_folder = r"C:\raj\RnD\exp-rag-001\mathtutor\books"  # Change this to your folder path
    db_location = "./chroma_maths_pdf_db"

    pdf_files = find_pdf_files(pdf_folder)
    if not pdf_files:
        print("No PDF files found.")
        exit(1)

    print(f"Found {len(pdf_files)} PDF files. Loading and embedding...")

    documents = load_documents_from_pdfs(pdf_files)
    # Filter out empty documents
    documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not documents:
        print("No non-empty PDF documents found.")
        exit(1)

    embeddings = OllamaEmbeddings(model="embeddinggemma:300m")

    # Always rebuild the vector store for fresh data
    if os.path.exists(db_location):
        import shutil
        shutil.rmtree(db_location)

    vector_store = Chroma(
        collection_name="local_pdf_data",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    vector_store.add_documents(documents=documents)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = OllamaLLM(model="gemma4:e4b")

    prompt = ChatPromptTemplate.from_template(
        """Answer the user's question based on the provided context.
        Context: {context}
        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # while True:
    #     question = input("Enter your question (or type 'quit' to exit): ")
    #     if question.strip().lower() in ['quit', 'exit']:
    #         print("Exiting.")
    #         break
    #     response = rag_chain.invoke({"input": question})
    #     print(response["answer"])
