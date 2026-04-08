import os
import uuid
import base64
from pathlib import Path
import gradio as gr
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader

MODEL_NAME = "gemma4:e4b"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# RAG Configuration - Use paths relative to repository root
REPO_ROOT = Path(__file__).parent.parent  # Go up from mathtutor/ to repo root
RAG_DB_LOCATION = str(REPO_ROOT / "chroma_maths_pdf_db")
RAG_COLLECTION_NAME = "local_pdf_data"
PDF_FOLDER = str(REPO_ROOT / "mathtutor" / "books")

SYSTEM_PROMPT = """
You are a patient math tutor.

Rules:
- Teach math step by step in simple language.
- Do not skip important algebra or arithmetic steps.
- If the notebook image is unclear, say exactly what is unreadable.
- If a question refers to the uploaded page, use the image context first.
- Keep explanations concise but educational.
- End with one short follow-up practice question when appropriate.
- If the student asks a follow-up question, answer in context of the earlier page and chat.
"""

# Global RAG components (initialized lazily)
_rag_retriever = None
_rag_chain = None

def initialize_rag():
    """Initialize RAG system with PDF knowledge base"""
    global _rag_retriever, _rag_chain

    if _rag_retriever is not None:
        return True  # Already initialized

    try:
        # Check if vector DB exists
        print(f"[RAG] Checking for database at: {RAG_DB_LOCATION}")
        if not os.path.exists(RAG_DB_LOCATION):
            print(f"[RAG] Database not found at {RAG_DB_LOCATION}")
            return False

        print(f"[RAG] Database found! Loading embeddings model...")
        embeddings = OllamaEmbeddings(model="embeddinggemma:300m")

        print(f"[RAG] Connecting to Chroma vector store...")
        vector_store = Chroma(
            collection_name=RAG_COLLECTION_NAME,
            persist_directory=RAG_DB_LOCATION,
            embedding_function=embeddings
        )

        print(f"[RAG] Creating retriever...")
        _rag_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        print(f"[RAG] Setting up LLM and RAG chain...")
        llm = OllamaLLM(model=MODEL_NAME)

        prompt = ChatPromptTemplate.from_template(
            """You are a patient math tutor. Answer the student's question using the provided context from math textbooks.

            Context from textbooks:
            {context}

            Student's question: {input}

            Provide a clear, step-by-step explanation suitable for students."""
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        _rag_chain = create_retrieval_chain(_rag_retriever, document_chain)

        print(f"[RAG] ✅ Initialization complete!")
        return True
    except Exception as e:
        print(f"[RAG] ❌ Error initializing RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_rag_context(question):
    """Retrieve relevant context from knowledge base"""
    global _rag_retriever

    if _rag_retriever is None:
        return None, []

    try:
        retrieved_docs = _rag_retriever.get_relevant_documents(question)
        if not retrieved_docs:
            return None, []

        # Format context for display
        context_items = []
        for i, doc in enumerate(retrieved_docs[:3]):  # Show top 3
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:200].strip()
            context_items.append({
                "source": os.path.basename(source) if source != "Unknown" else "Unknown",
                "page": page,
                "snippet": snippet
            })

        # Combine content for RAG
        full_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        return full_context, context_items
    except Exception as e:
        print(f"Error retrieving RAG context: {e}")
        return None, []

def save_uploaded_image(image):
    if image is None:
        return None
    ext = ".png"
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    image.save(file_path)
    return str(file_path)


def save_uploaded_audio(audio_path):
    """Save audio file to uploads directory"""
    if audio_path is None:
        return None
    import shutil
    ext = Path(audio_path).suffix or ".wav"
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    shutil.copy(audio_path, file_path)
    return str(file_path)


def encode_audio_to_base64(audio_path):
    """Encode audio file to base64 string"""
    if audio_path is None:
        return None
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            return base64.b64encode(audio_data).decode()
    except Exception as e:
        print(f"Error encoding audio: {e}")
        return None


def build_messages(history, user_text, image_path=None, audio_path=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history:
        user_msg = {"role": "user", "content": turn["user"]}
        if turn.get("image_path"):
            user_msg["images"] = [turn["image_path"]]
        if turn.get("audio_path"):
            # audio_b64 = encode_audio_to_base64(turn["audio_path"])
            # if audio_b64:
            user_msg["audio"] = [turn["audio_path"]]
        messages.append(user_msg)
        messages.append({"role": "assistant", "content": turn["assistant"]})

    current_user = {"role": "user", "content": user_text}
    if image_path:
        current_user["images"] = [image_path]
    if audio_path:
        # audio_b64 = audio_path #encode_audio_to_base64(audio_path)
        # if audio_b64:
        current_user["images"] = [audio_path]
    messages.append(current_user)

    return messages

def chat_with_tutor(user_text, image, audio, use_rag, state):
    """Main chat function that handles text, image, audio input, and optional RAG"""
    # Save audio file if provided
    audio_path = save_uploaded_audio(audio) if audio is not None else None

    # Check if we have any input
    if not user_text and image is None and audio is None:
        return state, state, "", None, None, ""

    image_path = save_uploaded_image(image) if image is not None else None
    history = state if state is not None else []

    # Set default text if only media is provided
    effective_text = user_text.strip() if user_text else ""
    if not effective_text:
        if audio_path and image_path:
            effective_text = "Transcribe the audio and solve the math problem in the image."
        elif audio_path:
            effective_text = "Transcribe the audio and solve the math problem."
        elif image_path:
            effective_text = "Read this notebook page and explain the math problem step by step."

    # RAG Context Retrieval
    rag_context = None
    context_display = ""
    context_items = []

    if use_rag and effective_text:
        # Lazy initialization - only load RAG when first used
        if not initialize_rag():
            context_display = "❌ Failed to initialize Knowledge Base. Check console for errors."
            use_rag = False
        else:
            rag_context, context_items = get_rag_context(effective_text)

            if context_items:
                context_display = "### 📚 Retrieved from Knowledge Base:\n\n"
                for item in context_items:
                    context_display += f"**Source:** {item['source']} (Page {item['page']})\n"
                    context_display += f"*{item['snippet']}...*\n\n"
            else:
                context_display = "No relevant context found in knowledge base."

    # Build messages (regular path or RAG path)
    if use_rag and rag_context and not (image_path or audio_path):
        # Use RAG chain for pure text queries
        try:
            response = _rag_chain.invoke({"input": effective_text})
            assistant_text = response["answer"]
        except Exception as e:
            print(f"RAG error: {e}, falling back to regular chat")
            use_rag = False
            messages = build_messages(history, effective_text, image_path=image_path, audio_path=audio_path)
            response = ollama.chat(model=MODEL_NAME, messages=messages, options={"temperature": 0.2})
            assistant_text = response["message"]["content"]
    else:
        # Regular Ollama chat (supports images and audio)
        messages = build_messages(history, effective_text, image_path=image_path, audio_path=audio_path)

        # If RAG is enabled but we're using images/audio, enhance the prompt with context
        if use_rag and rag_context:
            messages[-1]["content"] = f"Reference material:\n{rag_context}\n\nStudent's question: {effective_text}"

        # Debug output
        if audio_path:
            print(f"\n[DEBUG] Sending audio to {MODEL_NAME}:")
            print(f"  - Audio file: {audio_path}")
            print(f"  - Text prompt: {effective_text}")
            print(f"  - Has image: {image_path is not None}")
            print(f"  - Using RAG: {use_rag}")

        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"temperature": 0.2}
        )

        assistant_text = response["message"]["content"]

    history.append({
        "user": effective_text,
        "assistant": assistant_text,
        "image_path": image_path,
        "audio_path": audio_path,
        "used_rag": use_rag and bool(context_items)
    })

    chat_view = []
    for item in history:
        user_label = item["user"]
        if item.get("image_path"):
            user_label = f"{user_label}\n[Notebook page uploaded]"
        if item.get("audio_path"):
            user_label = f"{user_label}\n[Audio question uploaded]"
        if item.get("used_rag"):
            user_label = f"📚 {user_label}\n[Answer enhanced with textbook knowledge]"
        chat_view.append({"role": "user", "content": user_label})
        chat_view.append({"role": "assistant", "content": item["assistant"]})

    return history, chat_view, "", None, None, context_display

def clear_chat():
    return [], [], "", None, None, ""

with gr.Blocks(title="Gemma Math Tutor") as demo:
    gr.Markdown("# 🎓 Notebook Math Tutor with Voice & Knowledge Base")
    gr.Markdown("**Upload a notebook page** 📷 | **Type or speak your question** 🎤 | **Search textbooks** 📚 | **Get step-by-step help!** 📝")

    # Check if RAG database is available (without loading models yet)
    rag_db_available = os.path.exists(RAG_DB_LOCATION)
    if rag_db_available:
        gr.Markdown("✅ **Knowledge Base Ready** - Your questions can be enhanced with content from math textbooks!")
    else:
        gr.Markdown("ℹ️ **Knowledge Base Not Available** - Run `rag010.py` first to index PDF textbooks.")

    state = gr.State([])

    chatbot = gr.Chatbot(height=400, show_label=False)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Scan notebook page")
            audio_input = gr.Audio(
                label="🎤 Speak your question (optional)",
                sources=["microphone", "upload"],
                type="filepath"
            )
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="✍️ Type your question (optional)",
                placeholder="Example: Solve this equation step by step",
                lines=4
            )

            # RAG Toggle
            rag_checkbox = gr.Checkbox(
                label="📚 Use Knowledge Base (search math textbooks)",
                value=rag_db_available,
                interactive=rag_db_available,
                info="Enable to search through uploaded textbooks for relevant context"
            )

            with gr.Row():
                send_btn = gr.Button("Ask tutor 📤", variant="primary", scale=2)
                clear_btn = gr.Button("Clear 🗑️", scale=1)

    # Context viewer (collapsible)
    with gr.Accordion("📖 Retrieved Context", open=False) as context_accordion:
        context_output = gr.Markdown("Context will appear here when using Knowledge Base...")

    send_btn.click(
        fn=chat_with_tutor,
        inputs=[text_input, image_input, audio_input, rag_checkbox, state],
        outputs=[state, chatbot, text_input, image_input, audio_input, context_output]
    )

    text_input.submit(
        fn=chat_with_tutor,
        inputs=[text_input, image_input, audio_input, rag_checkbox, state],
        outputs=[state, chatbot, text_input, image_input, audio_input, context_output]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[state, chatbot, text_input, image_input, audio_input, context_output]
    )

demo.launch()
