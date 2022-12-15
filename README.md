# Disaster Response Pipeline Project

### Description
The aim of this project was designing a Machine Learning algorithm that could label messages sent during natural disasters. This way, emergency services can read them in order of priority. 

There are 36 categories ("medical help", "food", "weather related", and many others). Once a message is processed, it is labeled with one or more categories. It can also be labeled with none, which would imply the message was considered non-relevant.

The database used to train and test the algorithm was provided by Appen (formally Figure8).

Once you execute the web app, you'll find two graphs:
- Distribution of Message Genres: It displays the proportion of messages received by direct messaging, taken from social media and taken from the news. It also shows the proportion of non-relevant messages in each genre.
- Distribution of Message Categories: It displays the frequency of each label, in order.

You'll also be able to input your own message, so it gets labeled by the algorithm.


### Instructions:
1. Download the files from GitHub and extract them.

2. Go to /models and extract "classifier.zip.001" (you will need 7-zip).

3. Using a command-line shell, navigate to /app.

4. Execute "pip install plotly pandas nltk flask joblib sqlalchemy scikit-learn".

5. Finally, go to /app and execute "python run.py".

6. The app will then be accessible from your browser. 


### Software requirements:
- Latest version of Python: https://www.python.org/downloads/
- 7-zip: https://www.7-zip.org/

### Observations:
By looking at the bar graphs, we can see that categories like "offer" or "shops" present a very low frequency. We even have "child alone", which was used 0 times.

This implies a clear underrepresentation of some categories. And thus, a poor classification of new messages related to those categories. 

One possible solution would be completing the training set with more examples of these categories. Anothe one would be merging similar categories.


### About:
This project was completed as a part of the Udacity Data Science Nanodegree by José Imedio in December 2022.
