# JETSON PYTHON 3.8 COMPATIBILITY PATCH MODULE

"""
MOTIVAZIONE DELLA PATCH:
Incompatibilità runtime tra la libreria Hugging Face Transformers (scritta per Python 3.9+) 
e l'ambiente Python 3.8 di sistema presente sul server unife Jetson.

L'aggiornamento di Python a livello di sistema non è praticabile per via dei vincoli 
sui driver CUDA. Questo modulo applica tecniche di "Monkey Patching" a runtime 
per risolvere due disallineamenti strutturali ed evitare crash trasversali.
"""

import typing
import warnings

# Disabilita gli avvisi di telemetria per mantenere pulito il terminale
warnings.filterwarnings("ignore")

# 1. Patch ForwardRef.__annotations__ per Python 3.8

# Hugging Face analizza i tipi usando l'oggetto interno typing.ForwardRef.
# Su Python 3.9+ tale oggetto possiede il dizionario __annotations__, assente su Python 3.8.
# Controllo se l'attributo manca e, in caso affermativo, lo iniettiamo come dizionario 
# vuoto {}. La libreria deduce l'assenza di annotazioni speciali e prosegue senza crash.
if not hasattr(typing.ForwardRef, '__annotations__'):
    typing.ForwardRef.__annotations__ = property(lambda self: {})

# 2. Patch BaseImageProcessor per evitare il leak di parametri testuali nei processori visivi

# Durante il preprocessing multimodale, i parametri riservati alla tokenizzazione del testo 
# (es. padding, max_length, truncation) possono sovrapporsi ed essere passati anche al motore 
# che analizza le immagini. Questa funzione intercetta la chiamata un istante prima dell'esecuzione, 
# rimuove (pop) i parametri testuali intrusi e passa all'elaboratore visivo solo i dati puliti.
try:
    from transformers.image_processing_utils import BaseImageProcessor
    _orig_image_call = BaseImageProcessor.__call__

    def _patched_image_call(self, images=None, **kwargs):
        for text_param in ["padding", "max_length", "stride", "truncation", "pad_to_multiple_of"]:
            kwargs.pop(text_param, None)
        return _orig_image_call(self, images=images, **kwargs)

    BaseImageProcessor.__call__ = _patched_image_call
except ImportError:
    pass
