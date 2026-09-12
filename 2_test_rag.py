import warnings
warnings.filterwarnings("ignore")

print("Avvio dello script in corso...", flush=True)

import chromadb
import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv 
from google import genai

# CONFIGURAZIONE API LLM GEMINI
# Cerca e carica automaticamente il file .env
load_dotenv()

# Recupera la chiave in modo sicuro
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Nessuna chiave trovata. Controlla il file .env!")

ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Configurazione parametri per ChromaDB
CHROMA_PATH = "vector_db"
COLLECTION_NAME = "knowledge_base"

def main():
    # FASE 1: TEST RICERCA SEMANTICA (RETRIEVER)

    # 1. Connessione a ChromaDB in sola lettura
    print(f"1. Connessione al database vettoriale in '{CHROMA_PATH}'...", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
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
        activation_fn=nn.Sigmoid(),  # Applica la sigmoide per normalizzare i punteggi tra 0 e 1
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

    # Filtraggio dei 6 chunk più pertinenti
    top_6 = ranked_results[:6]
    # top_3 = ranked_results[:3]


    # FASE 3: SINTESI LLM IN CLOUD (GEMINI)
    
    print("7. Costruzione del Prompt per il Cloud...", flush=True)

    # Concatenazione dei 6 migliori chunk in un'unica stringa formattata
    formatted_context = ""
    for idx, item in enumerate(top_6, start=1):
        source = item['metadata'].get('source', 'Sconosciuta')
        page = item['metadata'].get('page', 'Sconosciuta')
        chunk_id = item['chunk_id']
        text = item['doc']
        formatted_context += f"--- DOCUMENTO {idx} (ID Chunk: {chunk_id} | Fonte: {source} - Pag: {page}) ---\n{text}\n\n"

    # Ingegneria del Prompt (Prompt Engineering)
    prompt = f"""Sei un orientatore universitario rigoroso, preciso ed esaustivo.
    Il tuo compito è rispondere alla domanda dell'utente basandoti ESCLUSIVAMENTE sui documenti forniti nel CONTESTO sottostante.

    REGOLE TASSATIVE DI RISPOSTA:
    1. FEDELTÀ TESTUALE ASSOLUTA: Utilizza ESATTAMENTE i nomi delle figure professionali e dei settori presenti nel testo. NON inventare, sintetizzare con sinonimi o parafrasare i titoli (es. usa "Tecnici Web" se scritto così, NON "Specialista informatico").
    2. NESSUNA ALLUCINAZIONE: Se un'informazione o figura professionale non è esplicitamente citata nei documenti, NON includerla per alcun motivo.
    3. CITAZIONE FONTI: Alla fine di OGNI figura professionale o concetto riportato, inserisci la citazione esatta usando il file e la pagina indicati nell'intestazione del documento sorgente.
    Formato obbligatorio citazione: [Fonte: NOME_FILE, Pag. X]
    4. PAGINAZIONE: Considera la pagina indicata nell'intestazione di ciascun documento (es. Pag: 9) come l'UNICA pagina ufficiale valida per quel testo.
    5. STRUTTURA: Presenta i ruoli e gli sbocchi in un elenco puntato chiaro e descrittivo.

    CONTESTO DEI DOCUMENTI ESTRATTI:
    {formatted_context}

    DOMANDA DELL'UTENTE:
    {user_query}

    RISPOSTA:"""

    print("8. Invio richiesta a Gemini 3.6 Flash (Attendere...)", flush=True)
    
    # Generazione risposta impostando la 'temperature' a 0.1 per azzerare le allucinazioni
    response = ai_client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.1
        )
    )

    print("\n=========================================================")
    print("🤖 RISPOSTA GENERATA DALL'IA:")
    print("=========================================================")
    print(response.text)
    print("=========================================================\n")

if __name__ == "__main__":
    main()