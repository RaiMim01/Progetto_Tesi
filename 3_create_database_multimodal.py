try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import sys
import typing

# 1. Patch ForwardRef per compatibilità Python 3.8
if not hasattr(typing.ForwardRef, '__annotations__'):
    typing.ForwardRef.__annotations__ = property(lambda self: {})

import time
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disabilita gli avvisi del tokenizer

print("Avvio dello script in corso...", flush=True)

import chromadb
from chromadb.config import Settings
import pymupdf as fitz  # PyMuPDF
import torch
from transformers import AutoModel
from transformers.image_processing_utils import BaseImageProcessor

# 2. Patch su __call__ di BaseImageProcessor
_orig_image_call = BaseImageProcessor.__call__

def _patched_image_call(self, images=None, **kwargs):
    for text_param in ["padding", "max_length", "stride", "truncation", "pad_to_multiple_of"]:
        kwargs.pop(text_param, None)
    return _orig_image_call(self, images=images, **kwargs)

BaseImageProcessor.__call__ = _patched_image_call

# ==========================================
# CONFIGURAZIONE PARAMETRI MULTIMODALI
# ==========================================
DATA_PATH = "documents"
CHROMA_PATH = "vector_db"
MAX_PAGES_TO_TEST = float('inf')  # Nessun limite: elabora tutte le pagine


def get_alibaba_embedding(model, image=None, text=None):
    """Estrae gli embedding dal modello Alibaba AutoModel."""
    with torch.no_grad():
        if image is not None and text is not None:
            emb = model.get_fused_embeddings(texts=[text], images=[image])
        elif image is not None:
            emb = model.get_image_embeddings(images=[image])
        elif text is not None:
            emb = model.get_text_embeddings(texts=[text])
        else:
            raise ValueError("Fornire almeno un'immagine o un testo.")
            
        if hasattr(emb, "cpu"):
            emb = emb.cpu().tolist()
        elif hasattr(emb, "tolist"):
            emb = emb.tolist()
        return emb[0]


def main():
    print("Avvio elaborazione multimodale (Immagini PDF)...", flush=True)

    # 1. Inizializzazione Modello
    print("1. Caricamento del modello Alibaba Qwen2-VL...", flush=True)
    model_name = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    
    try:
        embedding_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        print("Modello Alibaba caricato con successo!", flush=True)
    except Exception as e:
        print(f"\nERRORE DI CARICAMENTO MODELLO: {e}")
        return

    # 2. Connessione a ChromaDB
    print(f"2. Connessione al Vector DB locale ('{CHROMA_PATH}')...", flush=True)
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PATH
    ))
    collection = client.get_or_create_collection(name="multimodal_kb", metadata={"hnsw:space": "cosine"})

    # OPTIMIZATION: Recupero di tutti gli ID già salvati per evitare ricalcoli
    existing_records = collection.get()
    existing_ids = set(existing_records["ids"]) if existing_records and "ids" in existing_records else set()
    print(f"Pagine già presenti nel database: {len(existing_ids)}", flush=True)

    # 3. Lettura PDF ed estrazione Immagini
    print("3. Scansione dei file PDF...", flush=True)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"Nessun PDF trovato nella cartella '{DATA_PATH}'.", flush=True)
        return

    processed_pages = 0
    skipped_pages = 0
    start_total = time.time()

    for filename in pdf_files:
        if processed_pages >= MAX_PAGES_TO_TEST:
            break
            
        filepath = os.path.join(DATA_PATH, filename)
        pdf_document = fitz.open(filepath)
        
        for page_num in range(len(pdf_document)):
            if processed_pages >= MAX_PAGES_TO_TEST:
                break
                
            page_id = page_num + 1
            custom_id = f"{filename}_p{page_id}_img"

            # Se la pagina è già nel database, la salta immediatamente
            if custom_id in existing_ids:
                skipped_pages += 1
                continue

            page = pdf_document.load_page(page_num)
            
            # Rendering della pagina in immagine
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            metadata = {"source": filename, "page": page_id, "type": "image"}
            
            print(f"   [NUOVA] Analisi visiva in corso: {filename} - Pagina {page_id}...", flush=True)
            
            # 4. Calcolo Embedding
            start_emb = time.time()
            vector = get_alibaba_embedding(embedding_model, image=img)
            emb_time = time.time() - start_emb
            
            # 5. Salvataggio in ChromaDB
            collection.upsert(
                ids=[custom_id],
                embeddings=[vector],
                metadatas=[metadata],
                documents=[f"Rappresentazione visiva di {filename}, pagina {page_id}"]
            )
            
            print(f"   -> Vettorizzazione completata in {emb_time:.2f} secondi.", flush=True)
            processed_pages += 1

    # Scrittura dei dati su disco
    if processed_pages > 0:
        client.persist()

    elapsed = time.time() - start_total
    print(f"\nESITO FINALE:", flush=True)
    print(f"- Nuove pagine vettorizzate: {processed_pages} (in {elapsed:.2f} s)", flush=True)
    print(f"- Pagine già esistenti saltate: {skipped_pages}", flush=True)
    print(f"- Elementi totali nella collezione 'multimodal_kb': {collection.count()}", flush=True)

if __name__ == "__main__":
    main()