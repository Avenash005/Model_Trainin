# ============================================================
# MILESTONE 1: Sentence Embeddings + Cosine Similarity Demo
# Using: sentence-transformers / all-MiniLM-L6-v2
# NO ChromaDB yet — just embeddings and similarity math
# ============================================================

# --- Import 1 ---
# numpy: numerical computing library.
# We use it to do the cosine similarity math (dot product, norms).
import numpy as np

# --- Import 2 ---
# SentenceTransformer: loads a pre-trained model that converts
# text → dense float vectors (embeddings). The model
# 'all-MiniLM-L6-v2' is fast, small, and very capable.
from sentence_transformers import SentenceTransformer

# ============================================================
# STEP 1: Load the model
# ============================================================
print("Loading model: all-MiniLM-L6-v2 ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")

# ============================================================
# STEP 2: Define 5 example sentences
# ============================================================
sentences = [
    "The king sat on his golden throne.",   # royalty / power
    "The queen wore a jewelled crown.",     # royalty / royalty
    "I love eating a crisp red apple.",     # fruit / food
    "Machine learning is transforming AI.", # tech
    "The cat chased the mouse around the garden.", # animals
]

# ============================================================
# STEP 3: Embed all sentences at once
# model.encode() returns a 2-D numpy array:
#   shape = (num_sentences, embedding_dim)
# ============================================================
embeddings = model.encode(sentences)

print(f"Number of sentences  : {len(sentences)}")
print(f"Embedding dimension  : {embeddings.shape[1]}")   # should be 384
print(f"Full embeddings shape: {embeddings.shape}\n")

# ============================================================
# HELPER: cosine similarity between two vectors
# cosine_sim = (A · B) / (||A|| * ||B||)
# Range: -1 (opposite) → 0 (unrelated) → 1 (identical meaning)
# ============================================================
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot   = np.dot(vec_a, vec_b)
    norm  = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / norm)

# ============================================================
# STEP 4: Compare pairs
# ============================================================
# Index map for readability
idx = {s: i for i, s in enumerate(sentences)}

king_vec  = embeddings[0]   # "The king sat on his golden throne."
queen_vec = embeddings[1]   # "The queen wore a jewelled crown."
apple_vec = embeddings[2]   # "I love eating a crisp red apple."

sim_king_queen = cosine_similarity(king_vec, queen_vec)
sim_king_apple = cosine_similarity(king_vec, apple_vec)

print("=" * 50)
print("COSINE SIMILARITY RESULTS")
print("=" * 50)
print(f"  king  <-> queen : {sim_king_queen:.4f}  (expect HIGH  — both royalty)")
print(f"  king  <-> apple : {sim_king_apple:.4f}  (expect LOW   — unrelated topics)")
print()

if sim_king_queen > sim_king_apple:
    print("✅ PASS: king is semantically CLOSER to queen than to apple.")
else:
    print("❌ UNEXPECTED: king scored closer to apple — check your model load.")

print("\n--- MILESTONE 1 COMPLETE ---")
print("Next step: add ChromaDB to store & query these embeddings.")
