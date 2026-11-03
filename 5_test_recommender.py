import os
import warnings
warnings.filterwarnings("ignore")

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "knowledge_base"

def main():
    print("=== TEST RECOMMENDER SYSTEM (TESTO) ===", flush=True)
    
    # 1. Connessione a ChromaDB
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"ERRORE: Collezione '{COLLECTION_NAME}' non trovata: {e}")
        return

    # 2. Selezione di un elemento di prova dal database
    all_records = collection.get(limit=1, include=["documents", "metadatas", "embeddings"])
    if not all_records["ids"]:
        print("Il database è vuoto. Popolalo prima con '1_create_database.py'.")
        return

    target_id = all_records["ids"][0]
    target_doc = all_records["documents"][0]
    target_meta = all_records["metadatas"][0]
    target_vector = all_records["embeddings"][0]

    print(f"\nDOCUMENTO TARGET SELEZIONATO:")
    print(f"- ID: {target_id}")
    print(f"- Fonte: {target_meta.get('source', 'N/D')} (Pag. {target_meta.get('page', 'N/D')})")
    print(f"- Estratto: {target_doc[:150]}...\n")

    # 3. Ricerca dei 3 elementi più simili tramite Similarità Coseno
    # Chiediamo 4 risultati perché il primo sarà il documento target stesso
    results = collection.query(
        query_embeddings=[target_vector],
        n_results=4,
        include=["documents", "metadatas", "distances"]
    )

    print("=========================================================")
    print("🔍 RACCOMANDAZIONI (Documenti più simili trovati):")
    print("=========================================================")

    rec_count = 1
    for doc, meta, dist, chunk_id in zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0], 
        results['ids'][0]
    ):
        # Escludiamo il documento di partenza
        if chunk_id == target_id:
            continue
            
        print(f"\n[Raccomandazione #{rec_count}]")
        print(f"- ID Chunk: {chunk_id}")
        print(f"- Fonte: {meta.get('source', 'N/D')} (Pagina {meta.get('page', 'N/D')})")
        print(f"- Distanza Coseno: {dist:.4f}")
        print(f"- Testo: {doc[:200]}...")
        rec_count += 1

if __name__ == "__main__":
    main()