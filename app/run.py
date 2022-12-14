import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('dataset', engine)
X = df['message']
Y = df[df.columns[4:]]

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # Data for graph 1
    genre_names = ['direct', 'social', 'news']
    
    df_genre_and_cats = df.iloc[:, 3:]
    df_genre_and_cats['sum'] = df_genre_and_cats.sum(axis=1)
    df_genre_and_cats = df_genre_and_cats[['genre', 'sum']]

    count = []
    for genre in genre_names:
        df_aux = df_genre_and_cats[df_genre_and_cats['genre'] == genre]
        count.append(df_aux[df_aux['sum'] != 0].shape[0])
        count.append(df_aux[df_aux['sum'] == 0].shape[0])
    
    df_table = pd.DataFrame({'Genre': ['direct', 'direct', 'social', 'social', 'news', 'news'],
                             'Categorized': ['yes', 'no', 'yes', 'no', 'yes', 'no'],
                             'Count': count}).sort_values(by="Categorized", ascending=False) 
    
    # Data for graph 2
    categories_counts = Y.sum().sort_values(ascending=False)
    categories_names = categories_counts.keys()
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    name = "Categorized",
                    x = df_table['Genre'],
                    y = df_table['Count'].iloc[:3]
                ),
                Bar(
                    name = "Non categorized",
                    x = df_table['Genre'],
                    y = df_table['Count'].iloc[3:]
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'barmode': 'stack',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()