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
Tokenizer.save_pretrained(english_text_tokenizer, 'tokenizer')
# Tokenize decoder input
_, vietnam_text_tokenizer = tokenize(viet_sentences)
Tokenizer.save_pretrained(vietnam_text_tokenizer, 'tokenizer')
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

model = load_model('final_epoch_wit_attention.h5')
model.summary()

