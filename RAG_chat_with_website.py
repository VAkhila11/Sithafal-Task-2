import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Initialize the Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index for vector storage
dim = 384  # Embedding dimension of 'all-MiniLM-L6-v2' model (should be 384)
index = faiss.IndexFlatL2(dim)  # Using L2 (Euclidean) distance

# Initialize Hugging Face's text generation pipeline (using GPT-2 or GPT-Neo)
generator = pipeline("text-generation", model="gpt2")

# Function to scrape a website and extract text content
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract textual content (e.g., paragraphs)
    text_content = ' '.join([p.text for p in soup.find_all('p')])  # Extract paragraphs as an example

    # Split text into chunks (for simplicity, split by sentences here)
    content_chunks = text_content.split('.')

    return content_chunks

# Function to embed and store website content
def embed_and_store_content(url):
    content_chunks = scrape_website(url)
    embeddings = model.encode(content_chunks)  # Generate embeddings for each chunk

    # Convert embeddings to numpy array of float32 (required by FAISS)
    embeddings = np.array(embeddings).astype('float32')

    # Add embeddings to FAISS index
    index.add(embeddings)  # Adds the embeddings directly to the FAISS index

# Example: Ingest data from multiple websites
websites = ["https://www.uchicago.edu/", "https://www.washington.edu/", "https://www.stanford.edu/", "https://und.edu/"]

for site in websites:
    embed_and_store_content(site)

# Function to handle user queries
def handle_query(query):
    # Convert the user query to an embedding
    query_embedding = model.encode([query])[0]

    # Convert query embedding to numpy array (float32)
    query_embedding = np.array(query_embedding).astype('float32')

    # Perform a similarity search to find the most relevant chunks
    D, I = index.search(query_embedding.reshape(1, -1), k=3)  # Retrieve top 3 results

    # Check if we have any valid indices
    if len(I[0]) == 0 or max(I[0]) < 0:
        return "No relevant content found for the query."

    # Return the distances and indices for debugging
    return D, I

# Example: Handle a query
user_query = "What is the mission of the University of Chicago?"
distances, indices = handle_query(user_query)

# If valid results are found, retrieve relevant content and generate response
if isinstance(indices, np.ndarray) and len(indices[0]) > 0:
    content_chunks = scrape_website(websites[0])  # Example content chunks
    relevant_chunks = []

    # Ensure the indices are valid before accessing
    for idx in indices[0]:
        if idx < len(content_chunks):  # Check if the index is within bounds
            relevant_chunks.append(content_chunks[idx])

    # Join the relevant chunks into a single context string
    context = "\n".join(relevant_chunks)

    # Generate response using Hugging Face's GPT-2 model
    prompt = f"Answer the following question based on the context provided:\n\n{context}\n\nQuestion: {user_query}"
    response = generator(prompt, max_length=150, num_return_sequences=1)

    print("Generated Response:", response[0]['generated_text'].strip())
else:
    print("No relevant content found for the query.")
