import os
import warnings
warnings.filterwarnings("ignore")

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "multimodal_kb"

def main():
    print("=== TEST RECOMMENDER SYSTEM (MULTIMODALE - ALIBABA QWEN2-VL) ===", flush=True)
    
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

    # 2. Selezione di una pagina di prova dal database
    all_records = collection.get(limit=1, include=["documents", "metadatas", "embeddings"])
    if not all_records["ids"]:
        print("Il database multimodale è vuoto. Esegui prima '3_create_database_multimodal.py'.")
        return

    target_id = all_records["ids"][0]
    target_meta = all_records["metadatas"][0]
    target_vector = all_records["embeddings"][0]

    print(f"\nPAGINA TARGET SELEZIONATA:")
    print(f"- ID: {target_id}")
    print(f"- File: {target_meta.get('source', 'N/D')}")
    print(f"- Pagina: {target_meta.get('page', 'N/D')}\n")

    # 3. Ricerca per similarità sui vettori visivo-testuali di Alibaba
    results = collection.query(
        query_embeddings=[target_vector],
        n_results=4,
        include=["documents", "metadatas", "distances"]
    )

    print("=========================================================")
    print("🖼️ PAGINE RACCOMANDATE (Pagine visivamente/semanticamente correlate):")
    print("=========================================================")

    rec_count = 1
    for meta, dist, page_id in zip(
        results['metadatas'][0], 
        results['distances'][0], 
        results['ids'][0]
    ):
        if page_id == target_id:
            continue
            
        print(f"\n[Raccomandazione #{rec_count}]")
        print(f"- ID Pagina: {page_id}")
        print(f"- File PDF: {meta.get('source', 'N/D')}")
        print(f"- Numero Pagina: {meta.get('page', 'N/D')}")
        print(f"- Distanza Coseno: {dist:.4f}")
        rec_count += 1

if __name__ == "__main__":
    main()