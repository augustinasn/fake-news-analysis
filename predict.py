import keras
import nltk
import re
import math

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from keras.preprocessing import text, sequence
from keras.models import model_from_json

def predict(inp_text):
	# Prepare input:

	test_input = inp_text

	# test_input = """ Boris Epshteyn: “The expectation is that this will fail,
	# it will not succeed. You never know.” “These arguments [of Democrat Steve
	# Gallardo, the sole Democrat of the five-person Maricopa County Board of
	# Supervisors, and the Arizona Democratic Party] are weak.” “If they succeed
	# [with their Temporary Restraining Order, TRO] there is going to be a more
	# in-depth hearing next week.”
	# “We [patriots, deplorables] have to show righteous indignation … that this is
	# unacceptable … We cannot allow the courts, which refused to give rightful hearings
	# … we cannot allow for the court to manufacture ways to stand in the way.”
	# “Why don’t we all want to know the truth?’ “Why wouldn’t we want to know the truth?
	# The answer is obvious … here’s the answer, it’s about the old-fashioned, hard ballots…”
	# """

	### Parse html:
	soup = BeautifulSoup(test_input, "html.parser")
	first_text = soup.get_text()

	### Remove punctuation & special characters:
	first_text = re.sub('\[[^]]*\]', ' ', first_text)
	first_text = re.sub('[^a-zA-Z]',' ',first_text)  	# replaces non-alphabets with spaces
	first_text = first_text.lower() 				    # Converting from uppercase to lowercase

	### Remove stopwords:
	first_text = nltk.word_tokenize(first_text)
	first_text = [ word for word in first_text if not word in set(stopwords.words("english")) ]

	### Lemmatize:
	lemma = nltk.WordNetLemmatizer()
	first_text = [ lemma.lemmatize(word) for word in first_text ] 
	first_text = " ".join(first_text)

	### Tokenize:
	tokenizer = text.Tokenizer(num_words=10000)
	tokenizer.fit_on_texts([first_text])
	tokenized_train = tokenizer.texts_to_sequences([first_text])
	X_train = sequence.pad_sequences(tokenized_train, maxlen=300)


	print(X_train)

	# Load model:

	with open('model.json', 'r') as json_file:
		loaded_model_json = json_file.read()

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")

	print("Loaded model from disk")

	# Predict on test input:

	loaded_model.compile(optimizer=keras.optimizers.Adam(lr = 0.01),
		                 loss='binary_crossentropy',
		                 metrics=['accuracy'])

	try:
		result = loaded_model.predict(X_train)[0][0]
		result = round(result * 100)
	except:
		result = .0

	return result

	# score = loaded_model.evaluate(X, Y, verbose=0)
	# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))