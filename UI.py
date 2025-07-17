import os
import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

# Prevent accelerate from loading meta tensors without data
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

# Load environment variables
load_dotenv()

st.title("🎥 Video Transcript AI")


# Extract YouTube video ID
def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# User input for YouTube URL
url = st.text_input("Enter YouTube URL:")
video_id = extract_video_id(url) if url else None

if url:
    if video_id:
        st.success(f"Video ID: {video_id}")
    else:
        st.error("Invalid YouTube URL. Please check and try again.")

# Language selection
input_language = st.selectbox(
    "Select transcript language",
    ["en", "hi", "fr"],
)

transcript = None

# Fetch transcript
if video_id:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id=video_id, languages=[input_language]
        )
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        st.success("✅ Transcript fetched successfully.")
    except NoTranscriptFound:
        st.error("❌ No transcript found for this video in the selected language.")
    except TranscriptsDisabled:
        st.error("❌ Transcripts are disabled for this video.")
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")

# Proceed only if transcript exists
if transcript:
    # Split transcript into text chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(transcript)
    docs = [Document(page_content=text) for text in texts]

    # Embedding model initialization with HuggingFaceEmbeddings
    try:
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop()

    # Create FAISS vector store
    vector_store = FAISS.from_documents(docs, embedding)

    # Retriever setup
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # LLM setup
    LLM = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-alpha", task="chat-completion"
    )
    model = ChatHuggingFace(llm=LLM, temperature=0.2)

    # Prompt template for RAG
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
        input_variables=["context", "question"],
    )

    # User question input
    question = st.text_input("Enter your question:")

    # Format retrieved documents
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # RAG pipeline chain
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser

    # Process question if provided
    if question:
        with st.spinner("Generating answer..."):
            try:
                result = main_chain.invoke(question)
                st.write(result)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

else:
    st.info("Please enter a valid URL and ensure transcript is available to proceed.")
