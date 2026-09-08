import warnings
warnings.filterwarnings("ignore")

print("Avvio dello script in corso...", flush=True)

import chromadb
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "knowledge_base"

def main():
    # FASE 1: TEST RICERCA SEMANTICA (RETRIEVER)

    # 1. Connessione a ChromaDB in sola lettura
    print(f"1. Connessione al database vettoriale in '{CHROMA_PATH}'...", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"ERRORE: Impossibile trovare la collezione '{COLLECTION_NAME}'. Esegui prima '1_create_database.py'.")
        return

    print(f"   Database connesso. Documenti totali indicizzati: {collection.count()}", flush=True)

    # 2. Caricamento Modello di Embedding Bi-Encoder(PyTorch)
    print("2. Caricamento modello di embedding MODIFICA DA 'all-MiniLM-L6-v2' A 'paraphrase-multilingual-MiniLM-L12-v2'...", flush=True)
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 3. Definizione della domanda di prova e conversione in tensore PyTorch
    domanda_utente = "Quali sono gli sbocchi occupazionali e professionali previsti per un laureato in Ingegneria Elettronica ed Informatica?"
    print(f"\nDomanda di test: '{domanda_utente}'", flush=True)

    print("3. Conversione della domanda in Tensore PyTorch...", flush=True)
    query_tensor = embedding_model.encode(domanda_utente, convert_to_tensor=True)
    
    # 4. Retrieval/Recupero dei primi 10 chunk (Bi-Encoder)
    print("4. Esecuzione ricerca per similarità coseno su ChromaDB (k=10)...", flush=True)
    results = collection.query(
        query_embeddings=[query_tensor.cpu().tolist()],
        n_results=10
    )

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    # 5. FASE 2: Semantic Re-ranking con Cross-Encoder (PyTorch)

    print("5. Caricamento Cross-Encoder multilingue 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'...", flush=True)
    reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

    # Creazione delle coppie [Domanda, Chunk] da analizzare
    coppie_query_doc = [[domanda_utente, doc] for doc in documents]

    print("6. Calcolo attenzione incrociata (Cross-Attention)...", flush=True)
    scores_tensor = reranker.predict(
        coppie_query_doc,
        activation_fn=nn.Sigmoid(),  # Applica la sigmoide per normalizzare i punteggi tra 0 e 1
        convert_to_tensor=True,      # Mantiene l'output come Tensore PyTorch
        batch_size=8                 # Un batch_size contenuto
    )

    # Strutturazione dei dati per il riordinamento
    ranked_results = []
    for orig_idx, (doc, meta, dist, score) in enumerate(zip(documents, metadatas, distances, scores_tensor)):
        ranked_results.append({
            'original_rank': orig_idx + 1,
            'doc': doc,
            'metadata': meta,
            'chroma_dist': dist,
            'rerank_score': score.item()
        })

    # Ordinamento decrescente in base al punteggio del Cross-Encoder
    ranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

    # Filtraggio dei 3 chunk più pertinenti
    top_3 = ranked_results[:3]

    # Output
    print("\n=========================================================")
    print("--- RISULTATI DOPO RE-RANKING MATEMATICO (TOP 3 SELEZIONATI) ---")
    print("=========================================================")
    for i, item in enumerate(top_3, start=1):
        print(f"\n[POSIZIONE FINALE {i}] (Posizione originale in ChromaDB: #{item['original_rank']})")
        print(f"Punteggio Pertinenza (Cross-Encoder): {item['rerank_score']:.4f} / 1.0000")
        print(f"Distanza Coseno originale: {item['chroma_dist']:.4f}")
        print(f"Fonte: {item['metadata'].get('source', 'N/D')} - Pagina: {item['metadata'].get('page', 'N/D')}")
        print(f"Testo: {item['doc'][:200].strip()}...")
        print("-" * 65)

if __name__ == "__main__":
    main()