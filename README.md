# installation
pip install streamlit langchain langchain-google-genai langchain-community chromadb sentence-transformers pypdf python-dotenv google-generativeai

# What This Code Does

Loads a PDF file (college/course details).

Splits the text into small chunks for processing.

Creates embeddings using sentence-transformers/all-mpnet-base-v2.

Stores embeddings in Chroma vector database.

Uses Google Gemini (gemini-2.5-flash) model via LangChain.

Builds a RAG (Retrieval-Augmented Generation) system to answer from the PDF.

Adds safety filters to block harmful or explicit content.

Runs on Streamlit with chat-style interface.

Remembers previous chat messages during the session.

Make sure your .env file has your Gemini API key:

GEMINI_KEY=your_gemini_api_key_here


# Update your PDF path in the code:

file_path = r"C:\Users\YourName\Downloads\yourfile.pdf"


# Run the app:

streamlit run app.py

# How to Use

# Type your question in the Streamlit chat box, e.g.

“How many seats in BSc CS?”

“What is the fee for BSc Computer Science?”

The bot reads data from your PDF and gives the correct answer using Gemini AI.
