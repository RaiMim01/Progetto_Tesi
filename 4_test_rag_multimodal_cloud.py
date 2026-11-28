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

    # Chiediamo i risultati più simili in base al parametro configurabile
    print(f"4. Ricerca delle {TOP_K_IMAGES} pagine più pertinenti...", flush=True)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K_IMAGES
    )

    metadatas = results['metadatas'][0]

    # FASE 2: PREPARAZIONE IMMAGINI PER GEMINI

    print("5. Estrazione dinamica delle immagini dai PDF...", flush=True)

    # Creazione delle cartelle per i risultati e le immagini
    os.makedirs("eval_results/images", exist_ok=True)
    percorsi_immagini_salvate = []

    # Lista che conterrà un dizionario per ogni immagine con un testo descrittivo e l'immagine in Base64
    gemini_image_parts = []

    for idx, meta in enumerate(metadatas, 1):
        source = meta.get("source")
        page = meta.get("page")
        print(f"   -> Recupero immagine {idx}: {source} (Pagina {page})", flush=True)

        # Estrazione dell'immagine dal PDF e conversione in Base64 grazie alla funzione extract_image_from_pdf
        base64_img = extract_image_from_pdf(source, page)

        # Salvataggio immagine su disco per passarla al Giudice
        img_path = f"eval_results/images/img_{int(time.time())}_{idx}.jpg"
        with open(img_path, "wb") as f_img:
            f_img.write(base64.b64decode(base64_img))
        percorsi_immagini_salvate.append(img_path)
        

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

    # Stampa della risposta e le relative info su un file json per la valutazione automatica
    risultato_eval = {
        "domanda": user_query,
        "risposta": answer_text,
        "contesti_recuperati": percorsi_immagini_salvate, # Passiamo i percorsi delle immagini!
        "fonti": [f"{meta.get('source')} (Pag. {meta.get('page')})" for meta in metadatas],
        "pipeline": "multimodale",
        "modello": "cloud" # CAMBIA IN "local" NEL FILE LOCALE
    }
    
    file_path = f"eval_results/test_multi_cloud_{int(time.time())}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(risultato_eval, f, indent=4, ensure_ascii=False)
    
    print(f"Risultati salvati per la valutazione in: {file_path}")
   
   
if __name__ == "__main__":
    main()