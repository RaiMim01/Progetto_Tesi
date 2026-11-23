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
import torch
from transformers import AutoModel
from dotenv import load_dotenv
import urllib.request
import json

# CONFIGURAZIONE GEMINI API KEY E DATABASE

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
if not GEMINI_API_KEY:
    raise ValueError("Nessuna chiave trovata. Controlla il file .env!")

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "multimodal_kb"
DATA_PATH = "documents"

# Numero di Immagini
TOP_K_IMAGES = 4

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


def extract_image_from_pdf(filename, page_num):
    """Estrae un'immagine da un PDF specifico e la converte in Base64 per Gemini."""
    # Costruzione del percorso e apertura del file pdf con PyMuPDF
    filepath = os.path.join(DATA_PATH, filename)
    doc = fitz.open(filepath)
    # Recupero della pagina con conversione dell'indice partendo da 0 (PyMuPDF) rispetto a 1 (come salvato nel database)
    page = doc.load_page(page_num - 1)

    # Codice per il confronto tra modelli locali e cloud con immagini a risoluzione più bassa:
    # 1. Recupera le dimensioni originali della pagina in punti (pt)
    # rect = page.rect
    # max_dim = max(rect.width, rect.height)
    # 2. Calcola il fattore di scala per far sì che il lato lungo sia esattamente 768 px
    # (Un A4 standard è ~842 pt di altezza. 768 / 842 = ~0.91 di scala)
    # scale = 768.0 / max_dim
    # 3. Renderizza l'immagine applicando la matrice di scala
    # pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    # image_bytes = pix.tobytes("jpeg")

    # Codice per ottimizzazione dell'utilizzo di Gemini:
    # Renderizzazione della pagina in un'immagine (pixmap) con una scala di 1.5x (zoom) per una migliore qualità
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    # Conversione dei pixel in bytes formato JPEG
    image_bytes = pix.tobytes("jpeg")
    # Doppia conversione: prima i bytes jpeg vengono codificati in Base64, poi convertiti in stringa UTF-8
    return base64.b64encode(image_bytes).decode("utf-8")


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

    user_query = "Quali lauree ci sono all'Università di Ferrara?"
    # user_query = "Quali sono gli esami del primo anno della laurea triennale in Ingegneria Elettronica ed Informatica?"
    # user_query = "Quali sono gli sbocchi occupazionali e professionali previsti per un laureato in Ingegneria Elettronica ed Informatica?"
    print(f"\nDomanda dell'utente: '{user_query}'", flush=True)

    print("3. Vettorizzazione della domanda...", flush=True)
    query_vector = get_alibaba_text_embedding(embedding_model, user_query)

    # Chiediamo i risultati più simili in base al parametro configurabile
    print(f"4. Ricerca delle {TOP_K_IMAGES} pagine più pertinenti...", flush=True)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K_IMAGES
    )

    metadatas = results['metadatas'][0]

    # FASE 2: PREPARAZIONE IMMAGINI PER GEMINI

    print("5. Estrazione dinamica delle immagini dai PDF...", flush=True)

    # Lista che conterrà un dizionario per ogni immagine con un testo descrittivo e l'immagine in Base64
    gemini_image_parts = []

    for idx, meta in enumerate(metadatas, 1):
        source = meta.get("source")
        page = meta.get("page")
        print(f"   -> Recupero immagine {idx}: {source} (Pagina {page})", flush=True)

        # Estrazione dell'immagine dal PDF e conversione in Base64 grazie alla funzione extract_image_from_pdf
        base64_img = extract_image_from_pdf(source, page)

        # Aggiungo un'etichetta testuale per Gemini per fargli capire quale immagine sta guardando
        gemini_image_parts.append({"text": f"--- IMMAGINE {idx} (Fonte: {source}, Pagina {page}) ---"})
        # Aggiungo i dati raw dell'immagine
        gemini_image_parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64_img
            }
        })

    # FASE 3: GENERAZIONE RISPOSTA CON LLM (GEMINI)

    print("\n6. Invio richiesta a Gemini 3.6 Flash...", flush=True)

    prompt_text = (
        f"Leggi attentamente il testo e le tabelle nell'immagine o nelle immagini sopra.\n"
        f"Rispondi in italiano in modo diretto, preciso e ben strutturato alla seguente domanda, "
        f"estraendo tutte le informazioni pertinenti (materie, CFU, semestri, requisiti o regole) "
        f"presenti nella pagina o nelle pagine:\n\n"
        f"{user_query}"
    )

    # Costruiamo il payload per Gemini: Prompt testuale + Immagini in Base64
    parts = [{"text": prompt_text}] + gemini_image_parts

    # Gemini 3.6 Flash
    # Definizione dell'url per Gemini a cui inviare la richiesta
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"

    # Definizione del payload JSON (dizionario) richiesto per la chiamata all'API di Gemini
    # Aggiunto maxOutputTokens per sicurezza e per simulare i limiti locali
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens":  8192
        }
    }

    start_gen = time.time()

    # Preparazione della richiesta HTTP con url, payload e headers (che specifica il tipo di chiamata json)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )

    prompt_tokens = 0
    gen_tokens = 0
    total_tokens = 0

    # Invio della richiesta a Gemini e gestione della risposta
    try:
        # urlopen() invia la richiesta HTTP POST
        # read().decode() legge la risposta e la coverte da byte a stringa formato JSON
        # json.loads() converte la stringa JSON in un dizionario Python
        # La risposta contiene la risposta generata da Gemini, che viene estratta dal dizionario e stampata
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            answer_text = res_data['candidates'][0]['content']['parts'][0]['text']       
            
            # Estrazione dei metadati per tracciare quanti token consuma ogni immagine su Gemini
            usage = res_data.get('usageMetadata', {})
            prompt_tokens = usage.get('promptTokenCount', 0)
            gen_tokens = usage.get('candidatesTokenCount', 0)
            total_tokens = usage.get('totalTokenCount', 0)

    # Il primo livello gestisce gli errori delle API HTTP      
    except urllib.error.HTTPError as err:
        print(f"\nERRORE HTTP {err.code}: {err.read().decode('utf-8')}")
        return
    # Il secondo livello gestisce gli altri errori che non vengono catturati nel primo livello
    except Exception as err:
        print(f"\nERRORE: {err}")
        return

    gen_time = time.time() - start_gen
    total_time = time.time() - start_total
    
    decode_speed = gen_tokens / gen_time if gen_time > 0 else 0

    print("\n" + "=" * 60)
    print("RISPOSTA GENERATA DA GEMINI (Multimodale):")
    print("=" * 60)
    print(answer_text)
    print("=" * 60)
    print(f"Token Input (Testo + Img): {prompt_tokens} | Token Output: {gen_tokens} | Totale Token: {total_tokens}")
    print(f"Tempo di richiesta (RTT + Gen): {gen_time:.2f} s | Tempo totale pipeline: {total_time:.2f} s")
    print(f"Velocità stimata (Incluso Network): {decode_speed:.2f} tok/s")

if __name__ == "__main__":
    main()