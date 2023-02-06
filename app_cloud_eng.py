import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

model_name="leobertolazzi/it5-small-dante"

st.header("danteIT5")

st_model_load = st.text('Loading style transfer model...')

@st.cache(allow_output_mutation=True)
def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    nltk.download('punkt')
    print("Model loaded!")
    return tokenizer, model

tokenizer, model = load_model()

st_model_load.text("")

st.markdown('This is a small app made with an [italian T5 model](https://huggingface.co/gsarti/it5-small) fine-tuned on the whole [*Divine Comedy*](https://en.wikipedia.org/wiki/Divine_Comedy) and its paraphrases in modern italian.')
st.markdown('You can have fun converting the style of your italian text to the one of Dante!')
st.markdown('The results may be far from perfect, but you can play with the *Generation settings* and try to obtain better results.')
st.markdown('All code is open sourced and [here](https://github.com/leobertolazzi/danteIT5) you can find the project repo.')

with st.sidebar:
    st.header("Generation settings")
    if 'num_titles' not in st.session_state:
        st.session_state.num_titles = 5
    def on_change_num_titles():
        st.session_state.num_titles = num_titles
    num_titles = st.slider("Number of sentences to generate", min_value=1, max_value=10, value=1, step=1, on_change=on_change_num_titles)
    if 'beams' not in st.session_state:
        st.session_state.beams = 6
    def on_change_beams():
        st.session_state.beams = beams
    beams = st.slider("Beams", min_value=0, max_value=12, value=6, step=1, on_change=on_change_beams)
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 0.
    def on_change_top_p():
        st.session_state.top_p = top_p
    top_p = st.slider("Top-p", min_value=0., max_value=1.00, value=0., step=0.05, on_change=on_change_top_p)

    st.markdown("Note: *Beams* and *Top-p* cannot be both set on non-zero values")

st.subheader("Text generation")

if 'text' not in st.session_state:
    st.session_state.text = ""
st_text_area = st.text_area('Input text', value=st.session_state.text, height=50)

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
        predicted_dante = [nltk.sent_tokenize(decoded_output.strip())[0] for decoded_output in decoded_outputs]
    elif top_p != 0. and beams == 0:
        outputs = model.generate(**inputs, max_length=128, do_sample=True, top_p=top_p)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predicted_dante = [nltk.sent_tokenize(decoded_output.strip())[0] for decoded_output in decoded_outputs]
    else:
        predicted_dante = []

    st.session_state.dante = predicted_dante

# generate title button
st_generate_button = st.button('Transfer style', on_click=transfer_style)

# title generation labels
if 'dante' not in st.session_state:
    st.session_state.dante = []

if len(st.session_state.dante) > 0:
    with st.container():
        for sent in st.session_state.dante:
            st.markdown("__"+ sent +"__")
