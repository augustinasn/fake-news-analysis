from flask import Flask, render_template, url_for, redirect, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from predict import *

import pandas as pd
import joblib
import spacy
import en_core_web_lg

from scipy.spatial.distance import jensenshannon

nlp = en_core_web_lg.load(disable=["tagger", "parser", "ner"])
nlp.max_length = 3000000

def spacy_tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]


class ArticleForm(FlaskForm):
    article = StringField("Text",
                           validators=[DataRequired()])

    submit = SubmitField("Submit")

app = Flask(__name__)
app.config["SECRET_KEY"] = "ayo"

@app.route("/", methods=["GET", "POST"])
def root():
	form = ArticleForm()

	if form.validate_on_submit():
		return redirect(url_for("query", article=form.article.data))
	
	return render_template("main.html", title="Main Page", form=form)


@app.route("/query", methods=["GET", "POST"])
def query():
	text = request.args["article"]
	truthness = predict(text)

	test_input = ""

	# """India recorded more than 330,000 new cases in 24 hours,
	# the health ministry said on Friday, the second consecutive day that the
	# country has set a global record for daily infections. The reported death
	# toll on Friday was more than 2,200, also a new high for the country.
	# About half of the cases in Delhi, the capital city of more than 20 million
	# people, are testing positive for a more contagious variant of the virus,
	# first detected last year in India, that is afflicting younger people, said
	# a health ministry official, Sujeet Singh.
	# It is unclear to what extent the variant is driving the surge in cases around
	# the country, with large gatherings of unmasked people and widespread neglect
	# of preventive measures also suspected.
	# """

	df = pd.read_csv("topic_modeling_df.csv")

	# Load vectorizer and model:
	lda = joblib.load("lda.csv")
	vectorizer = joblib.load("vectorizer.csv")
	data_vectorized = joblib.load("data_vectorized.csv")
	doc_topic_dist = pd.read_csv("doc_topic_dist.csv")



	def get_k_nearest_docs(doc_dist, k=5, get_dist=False):
	    '''
	    doc_dist: topic distribution (sums to 1) of one article
	    
	    Returns the index of the k nearest articles (as by Jensen–Shannon divergence in topic space). 
	    '''
	         
	    distances = doc_topic_dist.apply(lambda x: jensenshannon(x, doc_dist), axis=1)
	    k_nearest = distances[distances != 0].nsmallest(n=k).index
	    
	    if get_dist:
	        k_distances = distances[distances != 0].nsmallest(n=k)
	        return k_nearest, k_distances
	    else:
	        return k_nearest

	def recommendation(idx, k=5, plot_dna=False):
	    '''
	    Returns the title of the k papers that are closest (topic-wise) to the paper given by paper_id.
	    '''
	    
	    print(f"Input: {test_input[:300]}...")

	    recommended, dist = get_k_nearest_docs(doc_topic_dist.iloc[idx], k, get_dist=True)
	    recommended = df.iloc[recommended].copy()
	    recommended["similarity"] = 1 - dist
	    recommended["date"] = pd.to_datetime(recommended["date"])
	    recommended = recommended.sort_values(by="date", ascending=False)
	    recommended["date"] = recommended["date"].dt.strftime("%Y-%m-%d")

	    output = []

	    for t, d, s in recommended[['title', 'date', 'similarity']].values:
	        output.append({"title": t,
	        	           "date": d,
	        	           "similarity": round(s * 100)})

	    return output

	def find_related(inp_text):
		test_input = inp_text

		# Prepare data:
		input_vectorized = vectorizer.transform([test_input])
		input_topic_dist = lda.transform(input_vectorized)
		
		# Predict:
		doc_topic_dist.iloc[0] = input_topic_dist

		return recommendation(idx=0, k=10, plot_dna=True)

	related_articles = find_related(text)

	return render_template("query.html",
						   title="Query Results",
						   text=text,
						   truthness=truthness,
						   related_articles=related_articles)

if __name__ == "__main__":
    app.run(debug=True)