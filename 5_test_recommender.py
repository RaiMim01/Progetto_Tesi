import os
import warnings
warnings.filterwarnings("ignore")

# Disabilita l'invio anonimo di dati di utilizzo da parte di ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "knowledge_base"

def main():
    print("TEST RECOMMENDER SYSTEM (Modello Testuale) ===", flush=True)
    
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
    # collection.get(...): Recupera i record memorizzati nel database
    # limit=1 (int): Chiede a ChromaDB di estrarre un solo elemento 
    # include=[...] (list[str]): Forza ChromaDB a restituire non solo gli ID, 
    # ma anche i testi originali ("documents"), le informazioni descrittive ("metadatas") 
    # e i vettori matematici memorizzati ("embeddings")
    # all_records (dict): Dizionario contenente i risultati estratti sotto forma di liste parallele
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
    # collection.query(...): Esegue la ricerca k-NN (k-Nearest Neighbors) nello spazio vettoriale
    results = collection.query(
        query_embeddings=[target_vector],
        n_results=4,
        include=["documents", "metadatas", "distances"]
    )

    print("RACCOMANDAZIONI (Documenti più simili trovati):")
   
    rec_count = 1
    # zip(...): Funzione integrata di Python che prende più liste (testi, metadati, distanze e ID) 
    # e le "accoppia" elemento per elemento ad ogni iterazione del ciclo for.
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