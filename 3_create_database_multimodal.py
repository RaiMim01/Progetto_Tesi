import os
import hashlib
import time
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

print("Avvio dello script in corso...", flush=True)

import chromadb
import pymupdf as fitz  # PyMuPDF: libreria per estrarre immagini dai PDF
import torch
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURAZIONE PARAMETRI MULTIMODALI
# ==========================================
DATA_PATH = "documents"
CHROMA_PATH = "vector_db"
BATCH_SIZE = 1  # TASSATIVO a 1 per CPU i3 con modello da 2B parametri
MAX_PAGES_TO_TEST = 3  # Limite di sicurezza per il primo test

def main():
    print("Avvio elaborazione multimodale (Immagini PDF)...", flush=True)

    # 1. Inizializzazione Modello Multimodale
    print("1. Caricamento del modello Alibaba Qwen2-VL (potrebbe richiedere minuti)...", flush=True)
    # Se questo modello risulta troppo lento, sostituiscilo con: 'jinaai/jina-clip-v1'
    model_name = 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct'
    
    try:
        # trust_remote_code=True è spesso necessario per modelli custom come Qwen
        embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"\nERRORE DI CARICAMENTO MODELLO: {e}")
        print("Il modello Alibaba richiede librerie aggiuntive o è incompatibile con la CPU attuale.")
        print("Sostituisci 'model_name' alla riga 27 con 'jinaai/jina-clip-v1' e riprova.")
        return

    # 2. Connessione a ChromaDB (Nuova Collezione)
    print(f"2. Connessione al Vector DB locale ('{CHROMA_PATH}')...", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Creiamo una collezione separata per non mischiare testo e immagini
    collection = client.get_or_create_collection(name="multimodal_kb", metadata={"hnsw:space": "cosine"})

    # 3. Lettura PDF ed estrazione Immagini
    print("3. Scansione dei file PDF e conversione in immagini...", flush=True)
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"Nessun PDF in '{DATA_PATH}'.", flush=True)
        return

    processed_pages = 0
    start_total = time.time()

    for filename in pdf_files:
        if processed_pages >= MAX_PAGES_TO_TEST:
            break
            
        filepath = os.path.join(DATA_PATH, filename)
        pdf_document = fitz.open(filepath)
        
        for page_num in range(len(pdf_document)):
            if processed_pages >= MAX_PAGES_TO_TEST:
                print(f"\nRaggiunto il limite di sicurezza di {MAX_PAGES_TO_TEST} pagine.")
                break
                
            page = pdf_document.load_page(page_num)
            
            # Rendering della pagina in un'immagine (risoluzione ridotta per alleggerire la CPU)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Preparazione metadati e ID
            page_id = page_num + 1
            custom_id = f"{filename}_p{page_id}_img"
            metadata = {"source": filename, "page": page_id, "type": "image"}
            
            print(f"   Analisi visiva in corso: {filename} - Pagina {page_id}...", flush=True)
            
            # 4. Calcolo Embedding dell'Immagine
            start_emb = time.time()
            # Passiamo l'oggetto PIL Image al modello
            tensor = embedding_model.encode(img, convert_to_tensor=True)
            emb_time = time.time() - start_emb
            
            # 5. Salvataggio in ChromaDB
            collection.upsert(
                ids=[custom_id],
                embeddings=[tensor.cpu().tolist()],
                metadatas=[metadata],
                documents=[f"Rappresentazione visiva di {filename}, pagina {page_id}"]
            )
            
            print(f"   -> Vettorizzazione completata in {emb_time:.2f} secondi.", flush=True)
            processed_pages += 1

    elapsed = time.time() - start_total
    print(f"\nESITO: Processate {processed_pages} immagini in {elapsed:.2f} s.")
    print(f"Elementi totali nella collezione 'multimodal_kb': {collection.count()}")

if __name__ == "__main__":
    main()