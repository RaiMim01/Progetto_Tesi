import warnings
warnings.filterwarnings("ignore")

print("Avvio dello script in corso...", flush=True)

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "knowledge_base"

def main():
    print("--- FASE 1: TEST RICERCA SEMANTICA (RETRIEVER) ---", flush=True)

    # 1. Connessione a ChromaDB in sola lettura
    print(f"1. Connessione al database vettoriale in '{CHROMA_PATH}'...", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"ERRORE: Impossibile trovare la collezione '{COLLECTION_NAME}'. Esegui prima '1_create_database.py'.")
        return

    print(f"   Database connesso. Documenti totali indicizzati: {collection.count()}", flush=True)

    # 2. Caricamento Modello di Embedding (PyTorch)
    print("2. Caricamento modello di embedding 'all-MiniLM-L6-v2'...", flush=True)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Definizione della domanda di prova
    domanda_utente = "Quali sono le regole e la verbalizzazione degli esami?"
    print(f"\nDomanda di test: '{domanda_utente}'", flush=True)

    # 4. Converti la domanda in Tensore PyTorch
    print("3. Conversione della domanda in Tensore PyTorch...", flush=True)
    query_tensor = embedding_model.encode(domanda_utente, convert_to_tensor=True)
    
    # 5. Esegui la query su ChromaDB (Recupero dei primi 10 chunk, k=10)
    print("4. Esecuzione ricerca per similarità coseno su ChromaDB (k=10)...", flush=True)
    results = collection.query(
        query_embeddings=[query_tensor.cpu().tolist()],
        n_results=10
    )

    # 6. Visualizzazione dei risultati grezzi estratti
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    print(f"\n--- RISULTATI GREZZI ESTRATTI DA CHROMADB (Top {len(documents)}) ---")
    for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        # La distanza L2/Coseno: valori più bassi indicano maggiore similarità semantica
        print(f"\n[Risultato {idx}] | Distanza Coseno: {dist:.4f}")
        print(f"Fonte: {meta.get('source', 'N/D')} - Pagina: {meta.get('page', 'N/D')}")
        print(f"Testo: {doc[:150].strip()}...")
        print("-" * 50)

if __name__ == "__main__":
    main()