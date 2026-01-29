"""
ChromaDB Local Example
Demonstrates adding documents to a collection and querying them.
"""

import chromadb

# Initialize a persistent client (stores data locally in ./chroma_data)
client = chromadb.PersistentClient(path="./chroma_data")

# Create or get a collection
collection = client.get_or_create_collection(name="example_collection")

# Sample documents to add
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "JavaScript is the language of the web, used for frontend and backend.",
    "Rust provides memory safety without garbage collection.",
    "Go was designed at Google for building scalable systems.",
    "TypeScript adds static typing to JavaScript.",
]

ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

metadatas = [
    {"category": "language", "paradigm": "multi-paradigm"},
    {"category": "language", "paradigm": "multi-paradigm"},
    {"category": "language", "paradigm": "systems"},
    {"category": "language", "paradigm": "concurrent"},
    {"category": "language", "paradigm": "multi-paradigm"},
]

# Check if documents already exist
existing = collection.get(ids=ids)
if not existing["ids"]:
    print("=" * 60)
    print("INGESTING DATA")
    print("=" * 60)
    for i, doc in enumerate(documents):
        print(f"  [{ids[i]}] {doc}")
    
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"\nAdded {len(documents)} documents to collection '{collection.name}'")
else:
    print("=" * 60)
    print("DATA ALREADY EXISTS")
    print("=" * 60)
    print(f"Collection '{collection.name}' already has {len(existing['ids'])} documents")

# Query the collection
print("\n" + "=" * 60)
print("QUERYING")
print("=" * 60)

query_text = "memory safe programming language"
print(f"Query: \"{query_text}\"")

results = collection.query(
    query_texts=[query_text],
    n_results=3,
)

print("\n" + "=" * 60)
print("RESULTS (Top 3)")
print("=" * 60)

for i, (doc, meta, distance) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
)):
    print(f"\n{i + 1}. Distance: {distance:.4f}")
    print(f"   Document: {doc}")
    print(f"   Metadata: {meta}")

print("\n" + "=" * 60)
print(f"Data persisted in: ./chroma_data")
print("=" * 60)
