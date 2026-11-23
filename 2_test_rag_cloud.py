import os
import time
import warnings
warnings.filterwarnings("ignore")

# Disabilita l'invio anonimo di dati di utilizzo da parte di ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"

print("Avvio dello script in corso...", flush=True)

import chromadb
from chromadb.config import Settings
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv 
import google.generativeai as genai
import json
import urllib.request
import urllib.error

# CONFIGURAZIONE API LLM GEMINI
# Cerca e carica automaticamente il file .env
load_dotenv()

# Recupera la chiave in modo sicuro
# Sanificazione della chiave API da eventuali virgolette, spazi o newline letti dal file .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")

if not GEMINI_API_KEY:
    raise ValueError("Nessuna chiave trovata. Controlla il file .env!")

# Configurazione parametri per ChromaDB
CHROMA_PATH = "vector_db"
COLLECTION_NAME = "knowledge_base"
N_chunks = 6

def main():
    # Avvio cronometro per misurare il tempo totale della pipeline
    start_total = time.time()

    # FASE 1: TEST RICERCA SEMANTICA (RETRIEVER)

    # 1. Connessione a ChromaDB in sola lettura (Sintassi nativa ChromaDB 0.3.29)
    print(f"1. Connessione al database vettoriale in '{CHROMA_PATH}'...", flush=True)

    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))

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
    # user_query = "Quali lauree ci sono all'Università di Ferrara?"
    # user_query = "Quali sono gli esami del primo anno della laurea triennale in Ingegneria Elettronica ed Informatica?"
    user_query = "Quali sono gli sbocchi occupazionali e professionali previsti per un laureato in Ingegneria Elettronica ed Informatica?"
    print(f"\nDomanda di test: '{user_query}'", flush=True)

    print("3. Conversione della domanda in Tensore PyTorch...", flush=True)
    query_tensor = embedding_model.encode(user_query, convert_to_tensor=True)
    
    # 4. Retrieval/Recupero dei primi 10 chunk (Bi-Encoder)
    print("4. Esecuzione ricerca per similarità coseno su ChromaDB (k=10)...", flush=True)
    results = collection.query(
        query_embeddings=[query_tensor.cpu().tolist()],
        n_results=10
    )

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    chunk_ids = results['ids'][0]

    # FASE 2: Semantic Re-ranking con Cross-Encoder (PyTorch)

    print("5. Caricamento Cross-Encoder multilingue 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'...", flush=True)
    reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

    # Creazione delle coppie [Domanda, Chunk] da analizzare
    query_doc_pairs = [[user_query, doc] for doc in documents]

    print("6. Calcolo attenzione incrociata (Cross-Attention)...", flush=True)
    scores_tensor = reranker.predict(
        query_doc_pairs,
        activation_fct=nn.Sigmoid(),  # Applica la sigmoide per normalizzare i punteggi tra 0 e 1
        convert_to_tensor=True,      # Mantiene l'output come Tensore PyTorch
        batch_size=8                 # Un batch_size contenuto
    )

    # Strutturazione dei dati per il riordinamento
    ranked_results = []
    for orig_idx, (doc, meta, dist, score, chunk_id) in enumerate(zip(documents, metadatas, distances, scores_tensor, chunk_ids)):
        ranked_results.append({
            'original_rank': orig_idx + 1,
            'chunk_id': chunk_id,
            'doc': doc,
            'metadata': meta,
            'chroma_dist': dist,
            'rerank_score': score.item()
        })

    # Ordinamento decrescente in base al punteggio del Cross-Encoder
    ranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

    # Filtraggio dei chunk più pertinenti
    top_chunk = ranked_results[:N_chunks]
   
    # Stampa di confronto tra Top chunks Bi-Encoder e Top chunks Cross-Encoder
    print("\n--- TOP CHUNK BI-ENCODER (PRIMA DEL RE-RANKING) ---", flush=True)
    for rank in range(10):
        source = metadatas[rank].get('source', 'N/D')
        page = metadatas[rank].get('page', 'N/D')
        print(f"  {rank + 1}. ID: {chunk_ids[rank]} | Fonte: {source} (Pag. {page}) | Distanza: {distances[rank]:.4f}", flush=True)

    print("\n--- TOP CHUNK CROSS-ENCODER (PASSATI AL MODELLO) ---", flush=True)
    for rank, item in enumerate(top_chunk, start=1):
        source = item['metadata'].get('source', 'N/D')
        page = item['metadata'].get('page', 'N/D')
        print(f"  {rank}. ID: {item['chunk_id']} | Fonte: {source} (Pag. {page}) | Score: {item['rerank_score']:.4f} (Ex Bi-Encoder: #{item['original_rank']})", flush=True)
    print("-" * 52 + "\n", flush=True)

    # FASE 3: SINTESI LLM IN CLOUD (GEMINI)
    
    print("7. Costruzione del Prompt per il Cloud...", flush=True)

    # Concatenazione dei migliori chunk in un'unica stringa formattata
    formatted_context = ""
    for idx, item in enumerate(top_chunk, start=1):
        source = item['metadata'].get('source', 'Sconosciuta')
        page = item['metadata'].get('page', 'Sconosciuta')
        chunk_id = item['chunk_id']
        text = item['doc']
        formatted_context += f"--- DOCUMENTO {idx} (ID Chunk: {chunk_id} | Fonte: {source} | Pag: {page}) ---\n{text}\n\n"

    # Ingegneria del Prompt (Prompt Engineering)
    prompt = f"""Leggi attentamente i documenti forniti nel CONTESTO sottostante e rispondi in italiano alla DOMANDA DELL'UTENTE.

    REGOLE DI RISPOSTA:
    1. FEDELTÀ TESTUALE: Riporta ESATTAMENTE i dati, i termini e i requisiti presenti nel testo, senza inventare o aggiungere informazioni esterne.
    2. SOLO DATI PRESENTI: Basati esclusivamente sui documenti forniti. Se l'informazione non è presente, dichiaralo esplicitamente.
    3. CITAZIONE FONTI: Alla fine di ogni punto riportato, inserisci la citazione esatta usando file e pagina indicati nell'intestazione: [Fonte: NOME_FILE, Pag. X]
    4. STRUTTURA: Presenta le informazioni in un elenco puntato chiaro e ordinato.

    CONTESTO DEI DOCUMENTI ESTRATTI:
    {formatted_context}

    DOMANDA DELL'UTENTE:
    {user_query}

    RISPOSTA:"""

    print("8. Invio richiesta a Gemini 3.6 Flash (Attendere...)", flush=True)

    # Chiamata HTTP diretta all'endpoint REST ufficiale di Gemini 1.5 Flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Definizione del payload JSON (dizionario) richiesto per la chiamata all'API di Gemini
    # Aggiunto maxOutputTokens per allinearlo ai test eseguiti in locale
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192
        }
    }

    # Preparazione della richiesta HTTP con url, payload e headers (che specifica il tipo di chiamata json)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    # Variabili per i token (inizializzate a zero per sicurezza)
    prompt_tokens = 0
    gen_tokens = 0
    total_tokens = 0

    # Invio della richiesta a Gemini e gestione della risposta
    try:
        # Avvio cronometro per misurare il tempo di sola generazione in Cloud
        start_gen = time.time()

        # urlopen() invia la richiesta HTTP POST
        # read().decode() legge la risposta e la coverte da byte a stringa formato JSON
        # json.loads() converte la stringa JSON in un dizionario Python
        # La risposta contiene la risposta generata da Gemini, che viene estratta dal dizionario e stampata 
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            answer_text = res_data['candidates'][0]['content']['parts'][0]['text']
            
            # Estrazione dei metadati di utilizzo token per il piano gratuito e per confronto
            usage = res_data.get('usageMetadata', {})
            prompt_tokens = usage.get('promptTokenCount', 0)
            gen_tokens = usage.get('candidatesTokenCount', 0)
            total_tokens = usage.get('totalTokenCount', 0)

        gen_time = time.time() - start_gen
        total_time = time.time() - start_total

    # Il primo livello gestisce gli errori delle API HTTP
    except urllib.error.HTTPError as err:
        error_body = err.read().decode('utf-8')
        print(f"\nERRORE HTTP {err.code}: {err.reason}\nDettaglio errore da Google API:\n{error_body}")
        return
    # Il secondo livello gestisce gli altri errori che non vengono catturati nel primo livello
    except Exception as err:
        print(f"\nERRORE nella chiamata API a Gemini: {err}")
        return

    # Calcolo della velocità di decode stimata (Network + Generazione)
    decode_speed = gen_tokens / gen_time if gen_time > 0 else 0

    print("\n" + "=" * 60)
    print("RISPOSTA GENERATA DA GEMINI (Testuale):")
    print("=" * 60)
    print(answer_text)
    print("=" * 60)
    print(f"Token Input: {prompt_tokens} | Token Output: {gen_tokens} | Totale Token: {total_tokens}")
    print(f"Tempo di richiesta (RTT + Gen): {gen_time:.2f} s | Tempo totale pipeline: {total_time:.2f} s")
    print(f"Velocità stimata (Incluso Network): {decode_speed:.2f} tok/s")


if __name__ == "__main__":
    main()