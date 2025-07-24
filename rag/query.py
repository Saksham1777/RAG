from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from google.genai import types, errors
import os
from dotenv import load_dotenv
load_dotenv()


# Custom embedding class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# local model connection
model = "google_-_gemma-2-2b-it"
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="http://127.0.0.1:1234/v1"
)

embedding_model = SentenceTransformerEmbeddings("multi-qa-mpnet-base-dot-v1")
library = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Query
query = """For a solution formed by mixing liquids L and M, the vapour pressure of L plotted against 
the mole fraction of M in solution is shown in the following figure. Here xL and xM 
represent mole fractions of L and M, respectively, in the solution. The correct statement(s) 
applicable to this system is(are)"""
query_answer = library.similarity_search(query, k=2)
chunks = [doc.page_content for doc in query_answer]

# Re-rank
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(cross_encoder_model)
query_chunk_pairs = [[query, chunk] for chunk in chunks]
scores = reranker.predict(query_chunk_pairs, apply_softmax=True)
scored_chunks = list(zip(chunks, scores))
scored_chunks.sort(key=lambda x: x[1], reverse=True)

# print ranked chunks (rank, score)
print("\nRERANKED CHUNKS")
for rank, (chunk, score) in enumerate(scored_chunks, start=1):
    preview = chunk.replace("\n", " ")[:20]  # shorten long passages
    print(f"{rank:>2}. {score:.4f} | {preview}")
print(" rareank print over\n")

# Build context for the LLM
context = "\n\n".join([chunk for chunk, score in scored_chunks])

print("Context: \n", context, " \nContext Over\n")

# Prompt
prompt_template = """
You are an assistant who answers questions. Find answers to the question. Answer as precisely as possible using the context provided.
Task:
1. Read the numbered context snippets.
2. Decide which snippet(s) fully answer the question.
3. Copy or paraphrase only what is needed.
context:{context}
IMPORTANT - Answer strictly from these excerpts. If the answer is not present, say “Requested answer not found”.
"""
prompt_filled = prompt_template.format(context=context)

# API call
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_filled},
            {"role": "user", "content": query},
        ]
    )
    print(response.choices[0].message.content)
except errors.APIError as e:
    print(e.code)
    print(e.message)
