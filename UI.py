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
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "[Your API Token]"
# Load environment variables
load_dotenv()

st.set_page_config(page_title="Video Transcript AI", page_icon="▶️", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color:gray ; text-shadow: 2px 2px white; font-size: 50px;'>
        ▶️ Video Transcript AI ▶️
    </h1>
""",
    unsafe_allow_html=True,
)

# ------------------ Background wallpaper ------------------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://imgs.search.brave.com/Z9zfUJKMzm-kC4cGyyRyBjXwp5OUVBNmDfL-R_yzw44/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJjYXZlLmNv/bS93cC93cDEyNDE5/NTIzLmpwZw");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
    visibility: hidden;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


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


# ✅ Display video template function


# ------------------ YouTube URL Input with Thumbnail ------------------

# Layout with columns
col1, col2 = st.columns([3, 2])

with col1:
    url = st.text_input("Enter YouTube URL:")
    video_id = extract_video_id(url) if url else None
    # Language selection
    input_language = st.selectbox(
        "Select transcript language",
        ["en", "hi", "fr"],
    )
    if url:
        if video_id:
            st.success(f"Video ID: {video_id}")

with col2:
    video_id = extract_video_id(url) if url else None
    if video_id:
        # Display video thumbnail
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        st.image(thumbnail_url, caption="Video Thumbnail", use_container_width=True)

# Apply input box CSS styling
url_style = """
<style>
input[type="text"] {
    background-color: rgba(0, 0, 0, 0.5);
    border-radius: 10px;
    color: #fff;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 0.5px;
    width: 300;
}
</style>
"""
st.markdown(url_style, unsafe_allow_html=True)


# ------------------ Dropdown Styling ------------------
dropdown_style = """
<style>
div[data-baseweb="select"] > div {
    background-color: rgba(0, 0, 0, 0.5);
    border-radius: 10px;
    width: 300px;
}
div[data-baseweb="select"] span {
    color: #fff;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 0.5px;
}
</style>
"""

# ------------------ Label Styling ------------------
label_style = """
<style>
label {
    color: white !important;
    font-weight: bold;
    font-size: 20px;
    font-style: italic;
}
</style>
"""
st.markdown(label_style, unsafe_allow_html=True)
st.markdown(dropdown_style, unsafe_allow_html=True)

transcript = None
chunks = []
answer = ""

# Fetch transcript
if video_id:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id=video_id, languages=[input_language]
        )
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except NoTranscriptFound:
        st.error("❌ No transcript found for this video in the selected language.")
    except TranscriptsDisabled:
        st.error("❌ Transcripts are disabled for this video.")
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")

# Proceed only if transcript exists
if transcript:
    with st.spinner("Wait a Little..."):
        # Split transcript into text chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_text(transcript)
        chunks = [Document(page_content=text) for text in texts]

        # Embedding model initialization with HuggingFaceEmbeddings
        try:
            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    "use_auth_token": "[Your API Token]"
                },
            )
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            st.stop()

        # Create FAISS vector store
        vector_store = FAISS.from_documents(chunks, embedding)

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
                    answer = main_chain.invoke(question)
                    st.write(f"- **User Question Input**:{question}")
                    st.write(f"- **Generated Answer**:{answer}")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
else:
    st.info("Please enter a valid URL and ensure transcript is available to proceed.")

st.markdown("---")
st.caption("Made by Sachin Mishra")
