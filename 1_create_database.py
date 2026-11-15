import os
import re
import hashlib
import time
import warnings
warnings.filterwarnings("ignore")

# Disabilita l'invio anonimo di dati di utilizzo da parte di ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"

print("Avvio dello script in corso...", flush=True)

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

print("Librerie caricate con successo! Inizio lettura cartella...", flush=True)

# Configurazione percorsi e parametri
DATA_PATH = "documents"
CHROMA_PATH = "vector_db"
BATCH_SIZE = 32 

def main():
    # 1. Caricamento dei PDF tramite pypdf
    print("1. Lettura dei file PDF dalla cartella...", flush=True)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    if not documents:
        print(f"Nessun file PDF trovato in '{DATA_PATH}'. Aggiungi i tuoi file e riprova.", flush=True)
        return

    # Pulizia del testo da spazi e a-capo anomali
    print("   Pulizia del testo in corso...", flush=True)
    for doc in documents:
        doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
        # Converte il conteggio da 0 (default di pypdf) a 1 (pagina reale del PDF)
        if "page" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page"] + 1
    
    print(f"2. Caricate con successo {len(documents)} pagine totali dai PDF.", flush=True)

    # 2. Taglio del testo in chunk
    # Se i parametri di taglio vengono modificati -> Bisogna ricreare la collezione per non mischiare chunk di dimensioni diverse
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,      # Aumentato da 600 a 1100 per mantenere il contesto dei paragrafi
        chunk_overlap=200,     # Aumentato da 100 a 200 per evitare tagli sui margini
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
    
    # 3. Inizializzazione Vector Database Locale (ChromaDB)
    print(f"4. Connessione al Vector DB locale ('{CHROMA_PATH}')...", flush=True)
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))
    
    # creazione della collection "knowledge_base" con metadati per la Cosine Similarity 
    collection = client.get_or_create_collection(name="knowledge_base", metadata={"hnsw:space": "cosine"})

    # Recupera solo gli ID già salvati (senza scaricare i vettori) per evitare ricalcoli
    existing_records = collection.get(include=[])
    existing_ids = set(existing_records["ids"]) if existing_records and "ids" in existing_records else set()

    new_chunks = []
    new_ids = []
    for doc, cid in zip(chunks, all_ids):
        if cid not in existing_ids:
            new_chunks.append(doc)
            new_ids.append(cid)

    skipped_chunks = len(chunks) - len(new_chunks)
    print(f"  Chunk già presenti saltati: {skipped_chunks} | Nuovi chunk da elaborare: {len(new_chunks)}", flush=True)

    if not new_chunks:
        print(f"\nESITO: Nessun nuovo documento da indicizzare. Elementi totali in ChromaDB: {collection.count()}", flush=True)
        return

    # 4. Modello di Embedding (SentenceTransformers su PyTorch)
    print("5. Caricamento modello 'paraphrase-multilingual-MiniLM-L12-v2'...", flush=True)
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print(f"6. Calcolo tensori PyTorch e scrittura in ChromaDB...", flush=True)
    start_total = time.time()

    # Loop per il calcolo degli embedding e il salvataggio del database
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = [doc.metadata for doc in batch]
        
        batch_ids = new_ids[i : i + BATCH_SIZE]

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
        print(f"   Salvato batch {i // BATCH_SIZE + 1} / {(len(new_chunks) - 1) // BATCH_SIZE + 1} ({len(batch)} chunk)", flush=True)

    # Forza la scrittura su disco per DuckDB+Parquet
    client.persist()

    elapsed = time.time() - start_total
    
    print(f"\nESITO FINALE:")
    print(f"- Nuovi chunk vettorizzati: {len(new_chunks)} (in {elapsed:.2f} s)")
    print(f"- Chunk già esistenti saltati: {skipped_chunks}")
    print(f"- Elementi totali nella collezione 'knowledge_base': {collection.count()}")

if __name__ == "__main__":
    main()