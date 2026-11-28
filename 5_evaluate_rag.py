import os
import glob
import json
import time
import base64
import urllib.request
import urllib.error
import pandas as pd
from dotenv import load_dotenv
from openpyxl.styles import Alignment, Font, PatternFill

load_dotenv()
# Recupero chiave API per il servizio di inferenza esterno
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
if not OPENROUTER_API_KEY:
    raise ValueError("Nessuna chiave OPENROUTER_API_KEY trovata nel file .env!")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Modello giudice esterno (LLM-as-a-Judge) per mitigare il self-enhancement bias
JUDGE_MODEL = "nvidia/nemotron-3.5-super-vl:free"




# System prompt fisso: definisce il ruolo imparziale e mitiga il verbosity bias (Zheng et al., 2023)
SYSTEM_PROMPT = (
    "Sei un giudice accademico imparziale incaricato di valutare un sistema RAG "
    "(Retrieval-Augmented Generation) tramite single-answer grading. "
    "Sii rigoroso e oggettivo: non farti influenzare dalla lunghezza o dalla verbosità "
    "della risposta, ma premia esclusivamente accuratezza, fedeltà alle fonti e pertinenza. "
    "Ragiona sempre prima di assegnare qualsiasi punteggio e restituisci esclusivamente un JSON valido."
)

def chiama_giudice_esterno(istruzioni, pipeline, contesti):
    """Invia prompt e contesti (testuali o visivi) al giudice LLM forzando output JSON."""
    user_content = [{"type": "text", "text": istruzioni}]

    if pipeline == "testuale":
        for idx, chunk in enumerate(contesti, 1):
            user_content.append({
                "type": "text",
                "text": f"\n--- FONTE {idx} (Chunk Testuale) ---\n{chunk}"
            })
    else:
        # Codifica Base64 delle pagine PDF per la pipeline multimodale/visiva
        for idx, img_path in enumerate(contesti, 1):
            user_content.append({
                "type": "text",
                "text": f"\n--- FONTE {idx} (Pagina PDF come Immagine) ---"
            })
            if os.path.exists(img_path):
                with open(img_path, "rb") as f_img:
                    b64_data = base64.b64encode(f_img.read()).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                })

    # temperature=0.0 garantisce massima riproducibilità e determinismo nelle valutazioni
    # Rimosso response_format="json_object" perché con modelli reasoning multimodali su OpenRouter
    # può lasciare vuoto il campo content durante la fase di ragionamento
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": 8192
    }

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        raw_text = (res_data["choices"][0]["message"].get("content") or "").strip()
        # Estrae il blocco JSON {...} ignorando eventuali backtick o testo esterno
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx == -1 or end_idx == -1:
            raise ValueError("JSON vuoto o incompleto restituito dal modello giudice.")
        esito = json.loads(raw_text[start_idx:end_idx + 1])
        if not esito or not str(esito.get("motivo_fonti", "")).strip():
            raise ValueError("JSON vuoto o incompleto restituito dal modello giudice.")
        return esito


def valuta_singolo_test(dati):
    """Costruisce il prompt di valutazione e calcola le metriche RAG (Precision, Faithfulness, Relevancy)."""
    domanda = dati["domanda"]
    risposta = dati["risposta"]
    contesti = dati["contesti_recuperati"]
    pipeline = dati["pipeline"]
    num_fonti_totali = len(contesti)

    # Prompt strutturato: impone la spiegazione PRIMA del punteggio (Chain-of-Thought implicito)
    # per allinearsi al protocollo single-answer grading di Zheng et al. (2023)
    istruzioni = f"""Analizza la DOMANDA, le FONTI RECUPERATE (chunk testuali oppure immagini di pagine PDF) e la RISPOSTA GENERATA dal modello.

DOMANDA DELL'UTENTE:
{domanda}

RISPOSTA GENERATA DAL MODELLO:
{risposta}

NUMERO TOTALE DI FONTI RECUPERATE: {num_fonti_totali}

Compiti di valutazione (fornisci SEMPRE prima la motivazione e poi il punteggio):
1. ANALISI DELLE FONTI (Contextual Precision): Per ciascuna delle {num_fonti_totali} fonti fornite sotto, stabilisci se contiene informazioni utili per rispondere alla domanda (true) oppure se è fuori tema/inutile (false). Conta quante fonti sono pertinenti.
2. FEDELTÀ DELLA RISPOSTA (Faithfulness): Valuta da 0.0 a 1.0 quanto la risposta si attiene SOLO alle fonti fornite senza inventare dati (allucinazioni). (1.0 = totalmente fedele ai documenti, 0.0 = inventata).
3. PERTINENZA DELLA RISPOSTA (Answer Relevancy): Valuta da 0.0 a 1.0 quanto la risposta risponde in modo completo e diretto alla domanda dell'utente, senza premiare risposte inutilmente lunghe o ripetitive.

Restituisci ESCLUSIVAMENTE un oggetto JSON con questa struttura esatta:
{{
  "motivo_fonti": "<breve spiegazione in italiano su quali fonti/pagine erano giuste o sbagliate>",
  "dettaglio_fonti": [<lista di booleani true/false per ogni fonte in ordine>],
  "fonti_pertinenti_count": <intero da 0 a {num_fonti_totali}>,
  "motivo_faithfulness": "<breve spiegazione in italiano prima del voto>",
  "faithfulness_score": <float da 0.0 a 1.0>,
  "motivo_answer_relevancy": "<breve spiegazione in italiano prima del voto>",
  "answer_relevancy_score": <float da 0.0 a 1.0>
}}
"""

    esito = chiama_giudice_esterno(istruzioni, pipeline, contesti)

    # Normalizzazione dei conteggi per prevenire valori fuori scala restituiti dal modello
    fonti_giuste = int(esito.get("fonti_pertinenti_count", 0))
    fonti_giuste = max(0, min(fonti_giuste, num_fonti_totali))
    fonti_sbagliate = num_fonti_totali - fonti_giuste
    perc_fonti_giuste = round((fonti_giuste / num_fonti_totali) * 100, 1) if num_fonti_totali > 0 else 0.0

    return {
        "Fonti Totali Recuperate": num_fonti_totali,
        "Fonti Giuste (N)": fonti_giuste,
        "Fonti Sbagliate (N)": fonti_sbagliate,
        "% Fonti Giuste (Contextual Precision)": perc_fonti_giuste,
        "Motivo Fonti": esito.get("motivo_fonti", ""),
        "Faithfulness (0-1)": round(float(esito.get("faithfulness_score", 0.0)), 3),
        "Motivo Faithfulness": esito.get("motivo_faithfulness", ""),
        "Answer Relevancy (0-1)": round(float(esito.get("answer_relevancy_score", 0.0)), 3),
        "Motivo Answer Relevancy": esito.get("motivo_answer_relevancy", "")
    }


def main():
    files = sorted(glob.glob("eval_results/*.json"))
    if not files:
        print("Nessun file JSON trovato in 'eval_results/'.")
        return

    output_excel = "Risultati_Valutazione_RAG.xlsx"
    risultati_dettaglio = []
    file_gia_valutati = set()

    # Meccanismo di checkpointing: riprende l'esecuzione saltando i test già presenti nell'Excel
    if os.path.exists(output_excel):
        try:
            df_esistente = pd.read_excel(output_excel, sheet_name="Dettaglio_Domande")
            risultati_dettaglio = df_esistente.to_dict(orient="records")
            file_gia_valutati = set(df_esistente["File"].tolist())
            print(f"Riprendo dal file esistente: {len(file_gia_valutati)} test già valutati saltati.\n")
        except Exception:
            pass

    print(f"Trovati {len(files)} test totali. Modello Giudice attivo: {JUDGE_MODEL}\n")

    for i, file_path in enumerate(files, 1):
        nome_file = os.path.basename(file_path)
        if nome_file in file_gia_valutati:
            continue

        print(f"[{i}/{len(files)}] Valutazione di {nome_file}...", flush=True)
        with open(file_path, "r", encoding="utf-8") as f:
            dati = json.load(f)

        try:
            metriche = valuta_singolo_test(dati)
            riga = {
                "File": nome_file,
                "Pipeline": dati.get("pipeline", ""),
                "Modello": dati.get("modello", ""),
                "Domanda": dati.get("domanda", ""),
                "Risposta Generata": dati.get("risposta", ""),
                "Fonti Estratte": " | ".join(dati.get("fonti", [])),
                "Fonti Totali Recuperate": metriche["Fonti Totali Recuperate"],
                "Fonti Giuste (N)": metriche["Fonti Giuste (N)"],
                "Fonti Sbagliate (N)": metriche["Fonti Sbagliate (N)"],
                "% Fonti Giuste (Contextual Precision)": metriche["% Fonti Giuste (Contextual Precision)"],
                "Faithfulness (0-1)": metriche["Faithfulness (0-1)"],
                "Answer Relevancy (0-1)": metriche["Answer Relevancy (0-1)"],
                # Colonne predisposte per la validazione umana (calcolo Human-LLM Agreement)
                "VOTO UMANO Risposta (0-1)": None,
                "VOTO UMANO Fonti (0-1)": None,
                "Motivo Fonti": metriche["Motivo Fonti"],
                "Motivo Faithfulness": metriche["Motivo Faithfulness"],
                "Motivo Answer Relevancy": metriche["Motivo Answer Relevancy"]
            }
            risultati_dettaglio.append(riga)
            print(f"   -> Fonti giuste: {metriche['Fonti Giuste (N)']}/{metriche['Fonti Totali Recuperate']} ({metriche['% Fonti Giuste (Contextual Precision)']}%) | "
                  f"Faithfulness: {metriche['Faithfulness (0-1)']} | Relevancy: {metriche['Answer Relevancy (0-1)']}", flush=True)

            # Aggregazione statistica per Pipeline e Modello
            df_dettaglio = pd.DataFrame(risultati_dettaglio)
            df_riepilogo = df_dettaglio.groupby(["Pipeline", "Modello"], as_index=False).agg({
                "Domanda": "count",
                "Fonti Totali Recuperate": "sum",
                "Fonti Giuste (N)": "sum",
                "Fonti Sbagliate (N)": "sum",
                "% Fonti Giuste (Contextual Precision)": "mean",
                "Faithfulness (0-1)": "mean",
                "Answer Relevancy (0-1)": "mean"
            }).rename(columns={
                "Domanda": "N. Domande Testate",
                "% Fonti Giuste (Contextual Precision)": "% Media Fonti Giuste",
                "Faithfulness (0-1)": "Media Faithfulness",
                "Answer Relevancy (0-1)": "Media Answer Relevancy"
            }).round(3)

            # Esportazione incrementale su Excel con formattazione stilistica
            with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
                df_riepilogo.to_excel(writer, sheet_name="Riepilogo_Statistiche", index=False)
                df_dettaglio.to_excel(writer, sheet_name="Dettaglio_Domande", index=False)

                # --- FORMATTAZIONE FOGLIO 1: Riepilogo_Statistiche ---
                ws_riep = writer.sheets["Riepilogo_Statistiche"]
                ws_riep.freeze_panes = "A2"
                for col in ws_riep.columns:
                    col_letter = col[0].column_letter
                    header_val = str(col[0].value or "")
                    ws_riep.column_dimensions[col_letter].width = max(len(header_val) + 5, 16)
                    for cell in col:
                        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                        if cell.row == 1:
                            cell.font = Font(bold=True, color="FFFFFF")
                            cell.fill = PatternFill("solid", fgColor="2F5597")
                ws_riep.row_dimensions[1].height = 28

                # --- FORMATTAZIONE FOGLIO 2: Dettaglio_Domande ---
                ws_det = writer.sheets["Dettaglio_Domande"]
                ws_det.freeze_panes = "D2"

                larghezze_dettaglio = {
                    "File": 24,
                    "Pipeline": 14,
                    "Modello": 14,
                    "Domanda": 38,
                    "Risposta Generata": 50,
                    "Fonti Estratte": 28,
                    "Fonti Totali Recuperate": 14,
                    "Fonti Giuste (N)": 14,
                    "Fonti Sbagliate (N)": 14,
                    "% Fonti Giuste (Contextual Precision)": 18,
                    "Faithfulness (0-1)": 15,
                    "Answer Relevancy (0-1)": 16,
                    "VOTO UMANO Risposta (0-1)": 18,
                    "VOTO UMANO Fonti (0-1)": 18,
                    "Motivo Fonti": 42,
                    "Motivo Faithfulness": 42,
                    "Motivo Answer Relevancy": 42,
                }

                for col in ws_det.columns:
                    col_letter = col[0].column_letter
                    header_val = str(col[0].value or "")
                    ws_det.column_dimensions[col_letter].width = larghezze_dettaglio.get(header_val, 20)

                    for cell in col:
                        if cell.row == 1:
                            cell.font = Font(bold=True, color="FFFFFF")
                            if "VOTO UMANO" in header_val:
                                cell.fill = PatternFill("solid", fgColor="375623")
                            else:
                                cell.fill = PatternFill("solid", fgColor="2F5597")
                            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                        else:
                            if "VOTO UMANO" in header_val:
                                cell.fill = PatternFill("solid", fgColor="E2EFDA")
                                cell.alignment = Alignment(horizontal="center", vertical="center")
                            elif header_val in ["Domanda", "Risposta Generata", "Fonti Estratte",
                                                "Motivo Fonti", "Motivo Faithfulness", "Motivo Answer Relevancy"]:
                                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                            else:
                                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

                ws_det.row_dimensions[1].height = 32
                for r in range(2, ws_det.max_row + 1):
                    ws_det.row_dimensions[r].height = 75

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            print(f"   -> ERRORE HTTP {e.code} su {nome_file}: {err_msg}", flush=True)
        except Exception as e:
            print(f"   -> ERRORE su {nome_file}: {e}", flush=True)

        # Rate limiting per rispettare il tetto gratuito di 20 req/min su OpenRouter
        if i < len(files):
            time.sleep(8)

    print(f"\nFinito! Risultati salvati in {output_excel}")


if __name__ == "__main__":
    main()