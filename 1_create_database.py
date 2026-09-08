import os
import sys
import re
import hashlib
import time
import warnings
warnings.filterwarnings("ignore")

print("Avvio dello script in corso...", flush=True)

import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

print("Librerie caricate con successo! Inizio lettura cartella...", flush=True)

# Configurazione percorsi e parametri
DATA_PATH = "documents"
CHROMA_PATH = "vector_db"
BATCH_SIZE = 100 

def main():
    # 1. Caricamento dei PDF tramite pypdf
    print("1. Lettura dei file PDF dalla cartella...", flush=True)
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    if not documents:
        print(f"Nessun file PDF trovato in '{DATA_PATH}'. Aggiungi i tuoi file e riprova.", flush=True)
        return

    # Pulizia del testo da spazi e a-capo anomali
    print("   Pulizia del testo in corso...", flush=True)
    for doc in documents:
        doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
    
    print(f"2. Caricate con successo {len(documents)} pagine totali dai PDF.", flush=True)

    # 2. Taglio del testo in chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"3. Generati {len(chunks)} blocchi (chunks) di testo pronti per l'embedding.", flush=True)

    # Ciclo per la gederazione di ID univoci per ogni chunk
    chunk_counters = {}
    all_ids = []

    for doc in chunks:
        # Ricavo il nome del file dai metadati altrimenti ne creo uno con l'hash del contenuto del chunk
        source = doc.metadata.get("source")
        if source:
            filename = os.path.basename(source)
        else:
            # Creazione del nomefile basato sul codice hash del testo del chunk
            text_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:8]
            filename = f"anonimo_{text_hash}.pdf"
            doc.metadata["source"] = filename

        # Ricavo il numero d pagina
        page = doc.metadata.get("page", 0)

        # Creo la chiave per contare i chunk per pagina di un file
        key = f"{filename}_p{page}"
        chunk_counters[key] = chunk_counters.get(key, 0) + 1

        # Creazione e salvataggio dell'ID univoco
        custom_id = f"{filename}_p{page}_c{chunk_counters[key]}"
        all_ids.append(custom_id)
    
    # 3. Modello di Embedding (SentenceTransformers su PyTorch)
    print("4. Caricamento modello MODIFICA DA'all-MiniLM-L6-v2' A 'paraphrase-multilingual-MiniLM-L12-v2'...", flush=True)
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 4. Inizializzazione Vector Database Locale (ChromaDB)
    print(f"5. Connessione al Vector DB locale ('{CHROMA_PATH}')...", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # creazione della collection "knowledge_base" con metadati per la Cosine Similarity 
    collection = client.get_or_create_collection(name="knowledge_base", metadata={"hnsw:space": "cosine"})

    print(f"6. Calcolo tensori PyTorch e scrittura a lotti in ChromaDB...", flush=True)
    start_total = time.time()

    # Loop per il calcolo degli embedding e il salvataggio del database
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = [doc.metadata for doc in batch]
        
        batch_ids = all_ids[i : i + BATCH_SIZE]

        # Calcolo degli embedding con PyTorch e il modello di embedding
        tensors = embedding_model.encode(batch_texts, convert_to_tensor=True)

        # Stampa di verifica al primo ciclo
        if i == 0:
            print("\n--- DIMOSTRAZIONE TECNICA PYTORCH (Primo Batch) ---")
            print(f"Tipo di oggetto generato: {type(tensors)}")
            print(f"Forma della Matrice (Batch Shape): {tensors.shape}")
            print(f"Prime 5 coordinate del Chunk 0: {tensors[0][:5].tolist()}")
            print(f"Esempio di ID Intelligente: {batch_ids[0]}")
            print("---------------------------------------------------\n")

        # Conversione per il salvataggio nel database
        embeddings_list = tensors.cpu().tolist()

        # Salvataggio dei dati nel database ChromaDB con un update o una insert (upsert)
        collection.upsert(
            documents=batch_texts,
            embeddings=embeddings_list,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"   Salvato batch {i // BATCH_SIZE + 1} / {(len(chunks) - 1) // BATCH_SIZE + 1} ({len(batch)} chunk)", flush=True)

    elapsed = time.time() - start_total
    print(f"\nESITO: Processati {len(chunks)} chunk in {elapsed:.2f} s. Elementi totali in ChromaDB: {collection.count()}")

if __name__ == "__main__":
    main()