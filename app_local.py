import os
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

rootdir = 'model/'
for rootdir, dirs, files in os.walk(rootdir):
    for subdir in dirs:
        if 'checkpoint' in os.path.join(rootdir, subdir):
            model_dir = os.path.join(rootdir, subdir)

st.header("medievalIT5")

st_model_load = st.text('Loading style transfer model...')

@st.cache(allow_output_mutation=True)
def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    nltk.download('punkt')
    print("Model loaded!")
    return tokenizer, model

tokenizer, model = load_model()
st_model_load.text("")

st.markdown('Questa app utilizza un [modello T5 italiano](https://huggingface.co/gsarti/it5-base) al quale è stato fatto un fine-tuning su testi in italiano medievale.')
st.markdown("Qui puoi divertirti a convertire lo stile delle tue frasi dall'italiano contemporaneo a quello medievale!")
st.markdown('I risultati possono essere anche molto lontani dalla perfezione, ma puoi giocare con le  *Impostazioni* per provare ad ottenerne di migliori.')
st.markdown('La repository del progetto è disponibile [qui](https://github.com/leobertolazzi/medievalIT5).')
st.markdown("P.s. se non sai cosa scrivere prova con il testo di una canzone.")

with st.sidebar:
    st.header("Impostazioni")
    if 'num_titles' not in st.session_state:
        st.session_state.num_titles = 5
    def on_change_num_titles():
        st.session_state.num_titles = num_titles
    num_titles = st.slider("Numero di frasi da generare", min_value=1, max_value=10, value=1, step=1, on_change=on_change_num_titles)
    if 'beams' not in st.session_state:
        st.session_state.beams = 6
    def on_change_beams():
        st.session_state.beams = beams
    beams = st.slider("Beams", min_value=0, max_value=12, value=8, step=1, on_change=on_change_beams)
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 0.
    def on_change_top_p():
        st.session_state.top_p = top_p
    top_p = st.slider("Top-p", min_value=0., max_value=1.00, value=0., step=0.05, on_change=on_change_top_p)

    st.markdown("Nota: *Beams* e *Top-p* non possono avere entrambi valori diversi da zero.")
    st.markdown("Nota: se non sai quale valore di *Top-p* scegliere prova con 0.5.")

st.subheader("Generazione del testo")

if 'text' not in st.session_state:
    st.session_state.text = ""
st_text_area = st.text_area('Testo di input', value=st.session_state.text, height=50)

def transfer_style():
    st.session_state.text = st_text_area

    # number of output to generate
    inputs = [st_text_area for i in range(num_titles)]

    # tokenize text
    inputs = tokenizer(inputs, max_length=128, truncation=True, return_tensors="pt")

    # compute predictions based on generation setting selected
    if beams != 0 and top_p == 0.:
        outputs = model.generate(**inputs, max_length=128, do_sample=False, num_beams=beams, no_repeat_ngram_size=3)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predicted = [nltk.sent_tokenize(decoded_output.strip())[0] for decoded_output in decoded_outputs]
    elif top_p != 0. and beams == 0:
        outputs = model.generate(**inputs, max_length=128, do_sample=True, top_p=top_p)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predicted = [nltk.sent_tokenize(decoded_output.strip())[0] for decoded_output in decoded_outputs]
    else:
        predicted = []

    st.session_state.medieval = predicted

# generate title button
st_generate_button = st.button('Trasferisci stile', on_click=transfer_style)

# title generation labels
if 'medieval' not in st.session_state:
    st.session_state.medieval = []

if len(st.session_state.medieval) > 0:
    with st.container():
        for sent in st.session_state.medieval:
            st.markdown("__"+ sent +"__")
