# ============================================================
# MILESTONE 2: ChromaDB — Store & Query Embeddings
# Builds on Milestone 1: now we PERSIST vectors and do
# semantic search (find the most similar sentence to a query)
# ============================================================

# --- Import 1 ---
# numpy: same as before — used for cosine math if needed
import numpy as np

# --- Import 2 ---
# SentenceTransformer: loads the pre-trained MiniLM model
# that converts text -> 384-dim float vectors
from sentence_transformers import SentenceTransformer

# --- Import 3 ---
# chromadb: an open-source vector database that:
#   - stores embeddings (vectors)
#   - stores metadata (document text, labels, etc.)
#   - lets you query by similarity ("find me the nearest vectors")
import chromadb

# ============================================================
# STEP 1: Load the same model from Milestone 1
# (already cached locally — no download needed)
# ============================================================
print("Loading model: all-MiniLM-L6-v2 ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")

# ============================================================
# STEP 2: Create an in-memory ChromaDB client
# EphemeralClient() = lives in RAM only (no disk files yet)
# Great for testing; Milestone 3 will use PersistentClient()
# ============================================================
# chromadb.EphemeralClient(): creates a temporary in-memory DB
client = chromadb.EphemeralClient()

# A "collection" is like a table in a SQL DB, but for vectors.
# We give it a name so we can retrieve it later.
collection = client.create_collection(name="sentences")
print("ChromaDB collection created: 'sentences'\n")

# ============================================================
# STEP 3: Our 5 sentences (same as Milestone 1)
# ============================================================
sentences = [
    "The king sat on his golden throne.",
    "The queen wore a jewelled crown.",
    "I love eating a crisp red apple.",
    "Machine learning is transforming AI.",
    "The cat chased the mouse around the garden.",
]

# ============================================================
# STEP 4: Embed all sentences
# ============================================================
print("Embedding sentences...")
embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}  ({embeddings.shape[0]} sentences x {embeddings.shape[1]} dims)\n")

# ============================================================
# STEP 5: Add embeddings to ChromaDB
# collection.add() needs:
#   - embeddings: list of vectors (list of lists)
#   - documents:  the original text (stored alongside vectors)
#   - ids:        a unique string ID per document
# ============================================================
collection.add(
    embeddings=embeddings.tolist(),           # numpy -> plain list
    documents=sentences,                       # original text
    ids=[f"id_{i}" for i in range(len(sentences))]  # "id_0", "id_1", ...
)
print(f"Stored {collection.count()} documents in ChromaDB.\n")

# ============================================================
# STEP 6: Semantic Search — query the collection
# collection.query() finds the N most similar vectors
# to your query embedding using cosine distance internally
# ============================================================

queries = [
    "royalty and monarchy",          # should match king / queen
    "healthy snacks and fruits",     # should match apple
    "artificial intelligence",       # should match ML sentence
]

print("=" * 55)
print("SEMANTIC SEARCH RESULTS")
print("=" * 55)

for query_text in queries:
    # Embed the query into the same vector space
    query_vec = model.encode([query_text]).tolist()

    # Ask ChromaDB: give me the top-2 most similar documents
    results = collection.query(
        query_embeddings=query_vec,
        n_results=2,                  # top-2 matches
        include=["documents", "distances"]
    )

    print(f"\nQuery : \"{query_text}\"")
    print(f"{'Rank':<5} {'Distance':<10} {'Document'}")
    print("-" * 55)
    for rank, (doc, dist) in enumerate(
        zip(results["documents"][0], results["distances"][0]), start=1
    ):
        # ChromaDB returns L2 distance by default; lower = more similar
        print(f"  #{rank}  {dist:.4f}     {doc}")

print("\n--- MILESTONE 2 COMPLETE ---")
print("Next step: PersistentClient to save DB to disk + add more docs.")
