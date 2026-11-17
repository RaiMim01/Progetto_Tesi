import os
import time
import base64
import jetson_patches

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Avvio dello script in corso...", flush=True)

import chromadb
from chromadb.config import Settings
import pymupdf as fitz
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, Qwen2VLForConditionalGeneration
import gc

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "multimodal_kb"
DATA_PATH = "documents"
TOP_K_IMAGES = 1

def get_alibaba_text_embedding(model, text):
    """Calcola l'embedding di una stringa di testo usando il modello multimodale Alibaba Qwen2-VL."""
    # Disabilitazione del calcolo dei gradienti per risparmiare memoria
    with torch.no_grad():
        # Calcolo dell'embedding della query testuale passata
        emb = model.get_text_embeddings(texts=[text])
        # Conversione del tensore PyTorch in lista Python per la serializzazione
        if hasattr(emb, "cpu"):
            emb = emb.cpu().tolist()
        elif hasattr(emb, "tolist"):
            emb = emb.tolist()
        # Restituzione dell'embedding
        return emb[0]

def extract_pil_image_from_pdf(filename, page_num):
    """Estrae una pagina PDF e la restituisce direttamente come oggetto PIL Image."""
    filepath = os.path.join(DATA_PATH, filename)
    doc = fitz.open(filepath)
    page = doc.load_page(page_num - 1)
    # Scala 1.2x: ottimo compromesso tra leggibilità dei caratteri e consumo RAM
    pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def main():
    start_total = time.time()
    # FASE 1: RETRIEVAL (Ricerca per similarità)
    
    print(f"1. Connessione al database vettoriale in '{CHROMA_PATH}'...", flush=True)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    print("2. Caricamento modello di embedding multimodale Alibaba...", flush=True)
    model_name = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    embedding_model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )

    user_query = "Mostrami lo schema del piano di studi e le materie del primo anno."
    print(f"\nDomanda dell'utente: '{user_query}'", flush=True)

    print("3. Vettorizzazione della domanda...", flush=True)
    query_vector = get_alibaba_text_embedding(embedding_model, user_query)

    # Chiediamo i 3 risultati più simili (Top-3 è ideale per le immagini con Gemini)
    print("4. Ricerca delle pagine più pertinenti...", flush=True)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K_IMAGES 
    )

    metadatas = results['metadatas'][0]

    # LIBERAZIONE RAM: Scarichiamo il modello di embedding prima di caricare il generatore
    print("4. Rilascio memoria del modello di embedding...", flush=True)
    del embedding_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FASE 2: ESTRAZIONE IMMAGINI DAL PDF
    
    print(f"5. Estrazione delle top-{TOP_K_IMAGES} pagine PDF come immagini...", flush=True)
    pil_images = []
    message_content = []

    for idx, meta in enumerate(metadatas, 1):
        source = meta.get("source")
        page = meta.get("page")
        print(f"   -> Recupero immagine {idx}: {source} (Pagina {page})", flush=True)

        img = extract_pil_image_from_pdf(source, page)
        pil_images.append(img)

        message_content.append({"type": "text", "text": f"--- IMMAGINE {idx} (Fonte: {source}, Pagina {page}) ---"})
        message_content.append({"type": "image", "image": img})


    # FASE 3: GENERAZIONE RISPOSTA IN LOCALE CON QWEN2-VL
    
    print("\n6. Caricamento VLM generativo locale (Qwen/Qwen2-VL-2B-Instruct)...", flush=True)
    gen_model_name = "Qwen/Qwen2-VL-2B-Instruct"

    # Limitiamo i pixel per immagine (256-512 token visivi) per evitare Out-Of-Memory
    min_pixels = 512 * 28 * 28
    max_pixels = 768 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        gen_model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )

    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
        gen_model_name,
        torch_dtype=torch.float32,
        attn_implementation="eager"  # Bypass del kernel SDPA su CPU ARM Jetson
    )

    prompt_text = f"""Sei un assistente universitario esperto. Guarda attentamente le immagini delle pagine dei documenti ufficiali fornite sopra.

    COMPITI:
    1. Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sulle tabelle, schemi o testi visibili in queste immagini.
    2. Cita sempre la fonte specificando il nome del file e la pagina (es. [Fonte: Regolamento.pdf, Pagina 3]).
    3. Se l'informazione non è presente in nessuna delle immagini, di' chiaramente che i documenti non contengono la risposta.

    DOMANDA UTENTE: {user_query}
    RISPOSTA:"""

    message_content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": message_content}]

    print("7. Preparazione tensori multimodali e generazione risposta (potrebbe richiedere 1-3 minuti)...", flush=True)
    start_gen = time.time()

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text_prompt],
        images=pil_images,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,  # Generazione deterministica (più veloce e precisa)
            repetition_penalty=1.15,
            synced_gpus=False
        )


    # Taglia i token del prompt iniziale per tenere solo la risposta generata
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0]

    gen_time = time.time() - start_gen
    total_time = time.time() - start_total

    print("\n" + "=" * 60)
    print("RISPOSTA GENERATA IN LOCALE DA ALIBABA QWEN2-VL-2B:")
    print("=" * 60)
    print(answer_text)
    print("=" * 60)
    print(f"Tempo di sola generazione: {gen_time:.2f} s | Tempo totale pipeline: {total_time:.2f} s")

if __name__ == "__main__":
    main()