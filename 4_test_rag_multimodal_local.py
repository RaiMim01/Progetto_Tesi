import os
import io
import json
import time
import base64
import urllib.request
import urllib.error
import jetson_patches

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Avvio dello script in corso...", flush=True)

import chromadb
from chromadb.config import Settings
import pymupdf as fitz
from PIL import Image
import torch
from transformers import AutoModel
import gc

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "multimodal_kb"
DATA_PATH = "documents"
TOP_K_IMAGES = 2
# Endpoint locale del server C++ llama.cpp avviato sull'altro terminale (porta 18080)
LLAMA_SERVER_URL = "http://127.0.0.1:18080/v1/chat/completions"


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
    # Scala 1.5x per avere caratteri nitidi nelle tabelle prima del ridimensionamento
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# 768 o 640
def pil_image_to_base64(img, max_side=768):
    """Ridimensiona l'immagine per bilanciare nitidezza delle tabelle e velocità CPU."""
    img_copy = img.copy()
    # 768 px mantiene leggibili le tabelle A4 restando ben dentro i 2048 token di contesto
    img_copy.thumbnail((max_side, max_side), Image.LANCZOS)
    # Salva l'immagine in un buffer in memoria (senza scrivere file su disco)
    buffer = io.BytesIO()
    img_copy.save(buffer, format="JPEG", quality=85)
    # Converte i byte dell'immagine in stringa Base64 trasmissibile via JSON
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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

    # 15 DOMANDE TEST:
    # 1) Domande puntuali / Single-hop:
    # user_query="Quanti CFU e quante ore frontali prevede l'insegnamento di 'Geometria e algebra' al primo anno di Ingegneria Elettronica e Informatica?"
    # user_query="Entro quale data e con quanti CFU minimi uno studente del primo anno di Ingegneria Elettronica e Informatica può iscriversi al secondo anno senza risultare fuori corso?"
    # user_query="Quali tecnologie utilizza DocFlow AI per combinare ricerca vettoriale densa e ricerca lessicale sparsa nella sua pipeline di retrieval ibrido?"
    
    # 2) Domande su tabelle, diagrammi e infografiche:
    # user_query="In base allo schema architetturale illustrato nella Figura 1 di DocFlow AI, quali sono le tre macro-fasi della pipeline di Indexing (Batch / Offline)?"
    # user_query="Se uno studente di Ingegneria Elettronica e Informatica ha già acquisito i crediti per 'Architettura e sviluppo di servizi Internet', quale esame non può inserire nel piano di studi secondo la tabella delle incompatibilità?"
    user_query="Quali sono gli insegnamenti propedeutici obbligatori per poter sostenere l'esame di 'Analisi matematica II'?"

    # 3) Domande Multi-hop / Sintesi tra più pagine o documenti:
    # user_query="Quali sono tutti i corsi di laurea triennale dell'Università di Ferrara erogati prevalentemente a distanza secondo l'Allegato B al Regolamento Didattico?"
    # user_query="Quali sono i tre macro-scenari industriali in cui DocFlow AI è stato validato sul campo nella Sezione 4.3 e quale riduzione percentuale dei tempi di riconciliazione ordini è stata ottenuta nel settore manifatturiero?"
    # user_query="Come può uno studente assolvere gli Obblighi Formativi Aggiuntivi (OFA) assegnati al momento dell'immatricolazione in Ingegneria Elettronica e Informatica?"

    # 4) Domande di ragionamento, calcolo e disambiguazione:
    # user_query="Uno studente del secondo anno di Ingegneria Elettronica e Informatica ha superato tutti gli esami del primo anno tranne 'Lingua inglese: verifica delle conoscenze'. Può sostenere gli esami del terzo anno?"
    # user_query="Un lavoratore con esperienza professionale pregressa si iscrive alla triennale in Ingegneria Elettronica e Informatica e possiede anche una certificazione informatica ECDL: quanti CFU al massimo può farsi riconoscere per l'esperienza lavorativa e quanti per l'ECDL?"
    # user_query="Se uno studente sceglie nel terzo anno il curriculum di Ingegneria Informatica e vuole svolgere un tirocinio esterno più lungo sfruttando anche i crediti a libera scelta, quanti CFU complessivi può raggiungere per il tirocinio?"

    # 5) Domande "trappola" / Out-of-context:
    # user_query="A quanto ammonta l'importo esatto delle tasse universitarie annuali e qual è la soglia ISEE per l'esonero totale per gli studenti iscritti a Ingegneria Elettronica e Informatica a Ferrara?"
    # user_query="Qual è il punteggio esatto di accuratezza (es. F1-score o percentuale di Faithfulness) ottenuto da DocFlow AI sui benchmark RAG rispetto ad Adobe Acrobat AI Assistant?"
    # user_query="Chi è il docente titolare dell'insegnamento di 'Sistemi operativi' del secondo anno nell'anno accademico 2026/2027?"
    
    print(f"\nDomanda dell'utente: '{user_query}'", flush=True)

    print("3. Vettorizzazione della domanda...", flush=True)
    query_vector = get_alibaba_text_embedding(embedding_model, user_query)

    # Recupero delle pagine più simili dal database vettoriale (impostato da TOP_K_IMAGES)
    print("4. Ricerca delle pagine più pertinenti...", flush=True)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K_IMAGES
    )

    metadatas = results["metadatas"][0]

    # LIBERAZIONE RAM: Scarichiamo il modello di embedding prima di chiamare il generatore
    print("4b. Rilascio memoria del modello di embedding...", flush=True)
    del embedding_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FASE 2: ESTRAZIONE IMMAGINI DAL PDF

    print(f"5. Estrazione delle top-{TOP_K_IMAGES} pagine PDF come immagini...", flush=True)
    message_content = []

    # Creazione delle cartelle per i risultati e le immagini
    os.makedirs("eval_results/images", exist_ok=True)
    percorsi_immagini_salvate = []

    for idx, meta in enumerate(metadatas, 1):
        source = meta.get("source")
        page = meta.get("page")
        print(f"   -> Recupero immagine {idx}: {source} (Pagina {page})", flush=True)

        img = extract_pil_image_from_pdf(source, page)

        # Salvataggio immagine su disco per passarla al Giudice
        img_path = f"eval_results/images/img_local_{int(time.time())}_{idx}.jpg"
        img.save(img_path, "JPEG")
        percorsi_immagini_salvate.append(img_path)
        

        # Conversione dell'immagine PIL in Base64 per inviarla via API HTTP a llama-server
        img_base64 = pil_image_to_base64(img)

        message_content.append({
            "type": "text",
            "text": f"--- IMMAGINE {idx} (Fonte reale: {source}, Pagina {page}) ---"
        })
        # Formato standard OpenAI Vision per allegare immagini in Base64
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
        })

    # FASE 3: GENERAZIONE RISPOSTA IN LOCALE CON QWEN2-VL

    print("\n6. Invio richiesta al VLM generativo locale via llama-server (Qwen2-VL-2B Q4_K_M)...", flush=True)

    # Stringa con la fonte reale recuperata da ChromaDB (verrà aggiunta in fondo alla risposta)
    fonti_reali = ", ".join([f"{m.get('source')} - Pagina {m.get('page')}" for m in metadatas])

    prompt_text = (
        f"Leggi attentamente il testo e le tabelle nell'immagine o nelle immagini sopra.\n"
        f"Rispondi in italiano in modo diretto, preciso e ben strutturato alla seguente domanda, "
        f"estraendo tutte le informazioni pertinenti (materie, CFU, semestri, requisiti o regole) "
        f"presenti nella pagina o nelle pagine:\n\n"
        f"{user_query}"
    )

    message_content.append({"type": "text", "text": prompt_text})

    # Costruzione del payload JSON compatibile con l'API /v1/chat/completions di llama-server
    payload = {
        "model": "qwen2-vl-2b",
        "messages": [
            {"role": "user", "content": message_content}
        ],
        "max_tokens": 1024,     # Lunghezza massima della risposta generata
        "temperature": 0.1,      # Bassa temperatura per risposte deterministiche e fedeli al documento
        "repeat_penalty": 1.2,    # Impedisce al modello 2B di entrare in loop ripetitivi negli elenchi
        "repeat_last_n": 256,  # Controlla la ripetizione degli ultimi token generati
        "presence_penalty": 0.2     # Bassa temperatura per risposte deterministiche e fedeli al documento
    }

    print("7. Generazione risposta in corso ...", flush=True)
    start_gen = time.time()

    # Preparazione e invio della richiesta HTTP POST locale (nessun dato esce dal server)
    req = urllib.request.Request(
        LLAMA_SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            # Lettura ed estrazione del testo generato dalla risposta JSON del server
            result = json.loads(response.read().decode("utf-8"))
            answer_text = result["choices"][0]["message"]["content"].strip()
            finish_reason = result["choices"][0].get("finish_reason", "N/D")

            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            timings = result.get("timings", {})
            ttft_s = timings.get("prompt_ms", 0.0) / 1000.0
            real_tok_s = timings.get("predicted_per_second", 0.0)

            # Aggiunta automatica della fonte esatta recuperata dal database vettoriale
            answer_text = f"{answer_text}\n\n[Fonte: {fonti_reali}]"
    except urllib.error.URLError as e:
        print(f"\nERRORE DI CONNESSIONE a {LLAMA_SERVER_URL}: {e}")
        print("Verifica che llama-server sia attivo nell'altro terminale sulla porta 18080.")
        return

    gen_time = time.time() - start_gen
    total_time = time.time() - start_total

    print("\n" + "=" * 60)
    print("RISPOSTA GENERATA IN LOCALE DA QWEN2-VL-2B (GGUF):")
    print("=" * 60)
    print(answer_text)
    print("=" * 60)
    print(f"TTFT (Vision + Prefill): {ttft_s:.2f} s | Tempo di sola generazione: {gen_time:.2f} s | Tempo totale pipeline: {total_time:.2f} s")
    print(f"Token prompt (immagini + testo): {prompt_tokens} | Token generati: {completion_tokens} | Totale: {prompt_tokens + completion_tokens}/4096 | Velocità decode: {real_tok_s:.2f} tok/s | Stop: {finish_reason}")

# Stampa della risposta e le relative info su un file json per la valutazione automatica
    risultato_eval = {
        "domanda": user_query,
        "risposta": answer_text,
        "contesti_recuperati": percorsi_immagini_salvate,
        "fonti": [f"{meta.get('source')} (Pag. {meta.get('page')})" for meta in metadatas],
        "pipeline": "multimodale",
        "modello": "local"
    }

    file_path = f"eval_results/test_multi_local_{int(time.time())}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(risultato_eval, f, indent=4, ensure_ascii=False)

    print(f"Risultati salvati per la valutazione in: {file_path}")
    

if __name__ == "__main__":
    main()
