import sys
import warnings
warnings.filterwarnings("ignore")

print("1. Avvio dello script in corso...", flush=True)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("2. Librerie caricate con successo. Inizio lettura cartella...", flush=True)

# Percorso della cartella contenente i PDF
DATA_PATH = "documents"

def main():
    # 1. Caricamento dei PDF tramite pypdf
    print("Caricamento dei PDF in corso...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print(f"3. Nessun file PDF trovato nella cartella '{DATA_PATH}'. Aggiungine uno per provare!", flush=True)
        return
        
    print(f"3. Caricate con successo {len(documents)} pagine totali dai PDF.", flush=True)

    # 2. Taglio del testo in chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"4. Testo suddiviso in {len(chunks)} blocchi (chunks) pronti per l'IA.", flush=True)

    # 3. Anteprima del primo blocco
    print("\n--- ESEMPIO DI CHUNK ESTRATTO ---")
    print(chunks[25].page_content)
    print("---------------------------------")

if __name__ == "__main__":
    main()