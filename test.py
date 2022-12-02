import streamlit as st
import numpy as np
from keras.layers import LSTM, Input, TimeDistributed, Dense, Embedding, Dropout, Concatenate, Activation, Dot
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model


def tokenize(sentences):
    # Create tokenizer
    text_tokenizer = Tokenizer(filters='')
    # Fit texts
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer


en_filename = "en_sents.txt"
vi_filename = "vi_sents.txt"

data1 = open(en_filename, encoding='utf-8').read().strip().split("\n")
data2 = open(vi_filename, encoding='utf-8').read().strip().split("\n")

exclude = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
           '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
viet_sent = []
english_sent = []
for i in range(0, 254090):
    sd1 = data1[i].lower()
    sd = data2[i].lower()
    sd1 = ''.join([char for char in sd1 if char not in exclude])
    sd = ''.join([char for char in sd if char not in exclude])
    english_sent.append(sd1)
    viet_sent.append(sd)

X_train_decoder = []
Y_train_decoder = []
viet_sentences = []

for sentence in viet_sent:
    X_train_decoder.append("<start> " + sentence)
    Y_train_decoder.append(sentence + " <end>")
    viet_sentences.append("<start> " + sentence + " <end>")

english_text_tokenized, english_text_tokenizer = tokenize(english_sent)
# Tokenizer.save_pretrained(english_text_tokenizer, 'tokenizer')
# Tokenize decoder input
_, vietnam_text_tokenizer = tokenize(viet_sentences)
# Tokenizer.save_pretrained(vietnam_text_tokenizer, 'tokenizer')
viet_in_text_tokenized = vietnam_text_tokenizer.texts_to_sequences(X_train_decoder)
viet_out_text_tokenized = vietnam_text_tokenizer.texts_to_sequences(Y_train_decoder)

# Let's add 0 padding to the sentences, to make sure they are all the same length.
# That is, we must be sure that all Italian sentences have the same length as the
# longest English sentence and that all VietNam sentences have the same length
# as the longest English sentence
X_train_encoder = pad_sequences(english_text_tokenized, padding="post")
X_train_decoder = pad_sequences(viet_in_text_tokenized, padding="post")
Y_train_decoder = pad_sequences(viet_out_text_tokenized, padding="post")

# Let's check the length of the vocabulary
# Let's add one unit to size for 0 padding
english_vocab_size = len(english_text_tokenizer.word_index) + 1
vietnam_vocab_size = len(vietnam_text_tokenizer.word_index) + 1

# get the lenght of max italian/english sentence
max_english_len = X_train_encoder[0].shape[0]
max_vietnam_len = X_train_decoder[0].shape[0]

dich = st.text_input('Nhập vào câu cần dịch')

click = st.button('Translate')

model = load_model('final_epoch_wit_attention.h5')
model.summary()

if click:
    encoder_input = model.input[0]
    encoder_lstm_output, encoder_state_h, encoder_state_c = model.layers[4].output
    encoder_lstm_states = [encoder_state_h, encoder_state_c]
    encoder_model = Model(encoder_input,  # input encoder model
                          [encoder_lstm_output, encoder_state_h, encoder_state_c])  # output encoder model

    # decoder
    decoder_input = model.input[1]
    embeded_decoder = model.layers[3]
    embeded_decoder = embeded_decoder(decoder_input)
    decoder_state_h = Input(shape=(256), name="input_3")
    decoder_state_c = Input(shape=(256), name="input_4")
    decoder_state_inputs = [decoder_state_h, decoder_state_c]
    decoder_lstm = model.layers[5]
    decoder_output_lstm, state_h, state_c = decoder_lstm(embeded_decoder, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]

    # Attention mechanism in decoder
    encoder_out_as_input = Input(shape=(None, 256), name="input_5")
    dot_layer = model.layers[6]
    activation_dot_layer = model.layers[7]
    attention = dot_layer([decoder_output_lstm, encoder_out_as_input])
    attention = activation_dot_layer(attention)
    dot_layer2 = model.layers[8]
    context = dot_layer2([attention, encoder_out_as_input])
    conc_out = model.layers[9]
    conc_out = conc_out([context, decoder_output_lstm])

    # Decoder output
    dropout_out = model.layers[10]
    dropout_out = dropout_out(conc_out)
    decoder_dense = model.layers[11]
    decoder_outputs = decoder_dense(dropout_out)

    decoder_model = Model([decoder_input, encoder_out_as_input, decoder_state_inputs],  # input decoder model
                          [decoder_outputs] + decoder_states)  # output decoder model


    def pre_processing_sentece(sentence):
        # Tokenize words
        sentence_tokenized = english_text_tokenizer.texts_to_sequences([sentence])
        sentence_tokenized = pad_sequences(sentence_tokenized, max_english_len, padding="post")
        return sentence_tokenized


    def inference_with_attention(sentence):
        encoder_output, state_h, state_c = encoder_model.predict(sentence)

        # Define target word
        target_word = np.zeros((1, 1))
        # <start>:1 , <end>:2
        target_word[0, 0] = 1

        stop_condition = False
        # Define output sentence string
        sent = ''
        step_size = 0

        index_to_words = {idx: word for word, idx in vietnam_text_tokenizer.word_index.items()}
        while not stop_condition:

            # We are giving a target_word which represents <start> and encoder_states to the decoder_model
            # for the first step and the output at the previous step for the next steps after the first
            # If attention mechanism is active, we give as input the encoder output also
            output, state_h, state_c = decoder_model.predict([target_word, encoder_output, [state_h, state_c]])

            # As the target word length is 1. We will only have one time step
            # encoder_state_value = [state_h, state_c]
            # Find the word which the decoder predicted with max_probability
            output = np.argmax(output, -1)
            # The output is a integer sequence, to get back the word. We use our lookup table reverse_dict
            sent = sent + ' ' + str(index_to_words.get(int(output)))
            step_size += 1
            # If the max_length of the sequence is reached or the model predicted 2 (<end>) stop the model
            if step_size > max_vietnam_len or output == 2:
                stop_condition = True
            # Define next decoder input
            target_word = output.reshape(1, 1)

        return sent


    sentence_tokenized = pre_processing_sentece(dich)
    # st.write(sentence_tokenized)
    translated_sentence = inference_with_attention(sentence_tokenized)

    st.write(translated_sentence[:-5])
