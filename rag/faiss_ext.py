from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from PyPDF2 import PdfReader
import glob
from pathlib import Path

# Embedding class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, device="cpu")
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Embedding model
embedding_model = SentenceTransformerEmbeddings("all-mpnet-base-v2")

# Chunking function
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=25,
    separators=["\n\n", "\n", ".", " "]
)

# Read PDF
try:
    pdf_dir   = Path("D:\\Spanda\\ReRank&QDecomp\\data")
    pdf_paths = glob.glob(str(pdf_dir / "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDF files found in the 'data' folder.")
    print(f"Found {len(pdf_paths)} PDF(s)")
except Exception as e:
    print("PDF Reading error", e)

# List page wise
page_texts = []
for pdf_path in pdf_paths:
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                page_texts.append((text, page_num, Path(pdf_path).name))
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

print(f"PDFs split into {len(page_texts)} pages.\n")

# Chunk each page's text, preserving page number and filename in metadata
chunks_with_metadata = []
for text, page_num,filename in page_texts:
    # Split text into chunks page wise
    splitter = RecursiveCharacterTextSplitter()
    # Split and attach page number to each chunk
    page_chunks = splitter.split_text(text)
    for chunk in page_chunks:
        chunks_with_metadata.append({
            "content": chunk,
            "metadata": {
                "page": page_num, 
                "filename": filename}
        })

# Update metadata in the Document object creation
documents = [
    Document(
        page_content=chunk["content"],
        metadata=chunk["metadata"]
    )
    for chunk in chunks_with_metadata
]

# Doc information print
for doc in documents[:3]:
    print("doc metadata \n", doc.metadata)

# Chunk information print
print("number of chunks:", len(chunks_with_metadata))
for i, doc in enumerate(documents[:5], 1):
    print(f"Chunk {i}:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
    print()

# FAISS vector store
library = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)
library.save_local("faiss_index")
print("Faiss stored in 'faiss_index'")
