import os
import warnings
warnings.filterwarnings("ignore")

# Disabilita l'invio anonimo di dati di utilizzo da parte di ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"

print("Avvio dello script in corso...", flush=True)

import gc
import time
import chromadb
from chromadb.config import Settings
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import urllib.request
import urllib.error

# CONFIGURAZIONE SERVER LOCALE LLAMA.CPP (MINISTRAL-3-3B)
LLAMA_SERVER_URL = "http://127.0.0.1:18080/v1/chat/completions"

# Configurazione parametri per ChromaDB
CHROMA_PATH = "vector_db"
COLLECTION_NAME = "knowledge_base"
N_chunks=4

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
    print("2. Caricamento modello di embedding 'paraphrase-multilingual-MiniLM-L12-v2'...", flush=True)
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 3. Definizione della domanda di prova e conversione in tensore PyTorch

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

    # 6b. Rilascio memoria RAM dei modelli PyTorch prima della generazione su Jetson
    print("6b. Rilascio memoria RAM di Bi-Encoder e Cross-Encoder...", flush=True)
    del embedding_model, reranker, query_tensor, scores_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Filtraggio dei N chunks più pertinenti per rispettare la finestra di contesto
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


    # FASE 3: SINTESI LLM IN LOCALE (MINISTRAL-3-3B VIA LLAMA.CPP)
    
    print("7. Costruzione del Prompt per il modello locale...", flush=True)

    # Concatenazione dei migliori chunk in un'unica stringa formattata
    formatted_context = ""
    for idx, item in enumerate(top_chunk, start=1):
        source = item['metadata'].get('source', 'Sconosciuta')
        page = item['metadata'].get('page', 'Sconosciuta')
        chunk_id = item['chunk_id']
        text = item['doc']
        formatted_context += f"--- DOCUMENTO {idx} (ID Chunk: {chunk_id} | Fonte: {source} | Pag: {page}) ---\n{text}\n\n"

    # Ingegneria del Prompt (Prompt Engineering diretto senza role-playing per modelli Edge 3B)
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

    print("8. Invio richiesta a Ministral-3-3B locale via llama-server (Attendere...)", flush=True)

    # Definizione del payload JSON compatibile con l'API OpenAI di llama-server
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "repeat_penalty": 1.15,
        "repeat_last_n": 256,
        "max_tokens": 1024
    }

    # Preparazione della richiesta HTTP con url, payload e headers (che specifica il tipo di chiamata json)
    req = urllib.request.Request(
        LLAMA_SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    # Invio della richiesta a llama-server e gestione della risposta
    try:
        # Avvio cronometro per misurare il tempo di sola generazione locale
        start_gen = time.time()

        # urlopen() invia la richiesta HTTP POST
        # read().decode() legge la risposta e la converte da byte a stringa formato JSON
        # json.loads() converte la stringa JSON in un dizionario Python
        with urllib.request.urlopen(req, timeout=300) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            answer_text = res_data['choices'][0]['message']['content']
            finish_reason = res_data['choices'][0].get('finish_reason', 'N/D')
            usage = res_data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            timings = res_data.get('timings', {})
            ttft_s = timings.get('prompt_ms', 0.0) / 1000.0
            real_tok_s = timings.get('predicted_per_second', 0.0)


        gen_time = time.time() - start_gen
        total_time = time.time() - start_total
    
    # Il primo livello gestisce gli errori HTTP restituiti dal server
    except urllib.error.HTTPError as err:
        error_body = err.read().decode('utf-8')
        print(f"\nERRORE HTTP {err.code}: {err.reason}\nDettaglio errore da llama-server:\n{error_body}")
        return
    # Gestisce il caso in cui llama-server non sia ancora avviato sulla porta 8080
    except urllib.error.URLError as err:
        print(f"\nERRORE DI CONNESSIONE: Impossibile raggiungere llama-server su {LLAMA_SERVER_URL}. Verifica che sia avviato.\nDettaglio: {err.reason}")
        return
    # Il secondo livello gestisce gli altri errori che non vengono catturati nei livelli precedenti
    except Exception as err:
        print(f"\nERRORE nella chiamata a llama-server: {err}")
        return

    print("\n" + "=" * 60)
    print("RISPOSTA GENERATA IN LOCALE DA MINISTRAL-3-3B (GGUF):")
    print("=" * 60)
    print(answer_text)
    print("=" * 60)
    print(f"Tempo di sola generazione: {gen_time:.2f} s | Tempo totale pipeline: {total_time:.2f} s")
    tokens_sec = completion_tokens / gen_time if gen_time > 0 else 0
    print(f"Token prompt: {prompt_tokens} | Token generati: {completion_tokens} ({tokens_sec:.2f} tok/s) | Stop: {finish_reason}")
    print(f"TTFT (Prefill): {ttft_s:.2f} s | Tempo totale generazione: {gen_time:.2f} s | Tempo totale pipeline: {total_time:.2f} s")
    print(f"Token prompt: {prompt_tokens} | Token generati: {completion_tokens} | Totale: {prompt_tokens + completion_tokens}/4096 | Velocità decode: {real_tok_s:.2f} tok/s | Stop: {finish_reason}")

    # Stampa della risposta e le relative info su un file json per la valutazione automatica
    os.makedirs("eval_results", exist_ok=True)
    
    # Nel file 2_test_rag_local.py ricordati di cambiare "modello": "cloud" in "modello": "local"
    risultato_eval = {
        "domanda": user_query,
        "risposta": answer_text,
        "contesti_recuperati": [item['doc'] for item in top_chunk],
        "fonti": [f"{item['metadata'].get('source', 'N/D')} (Pag. {item['metadata'].get('page', 'N/D')})" for item in top_chunk],
        "pipeline": "testuale",
        "modello": "local" # CAMBIA IN "local" NEL FILE LOCALE
    }
    
    # Salva un file JSON univoco basato sul timestamp
    file_path = f"eval_results/test_txt_local_{int(time.time())}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(risultato_eval, f, indent=4, ensure_ascii=False)
    
    print(f"Risultati salvati per la valutazione in: {file_path}")

if __name__ == "__main__":
    main()