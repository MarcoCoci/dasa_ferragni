################ LIBRARIES ################

import pandas as pd
from transformers import pipeline
# import nltk # Only used to download stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm

import numpy as np
from scipy.special import softmax

from collections import Counter
import concurrent.futures

import plotly.express as px

tqdm.pandas()

################ DATASET ################

data = pd.read_csv('/Users/davidpaquette/Downloads/scraped_articles_instagram_reddit_linkedin_twitter.csv')
emoji_list = list(emoji.EMOJI_DATA.keys())

################ PREPROCESSING ################

### Check for nulls in 'text_content' 

nulls = data.isnull().sum()
nulls

### Nulls only in post_creator, likes, and date
### Don't need to be removed

# Saw many instances where 'text_content' was [deleted] - remove
data = data[data['text_content'] != '[deleted]']

### Split dataset based on language

# Initialize the 'language' column
data['language'] = pd.NA

# Language classfying model from huggingface
model_ckpt = "papluca/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model_ckpt)

# Function to iterate over texts and and output language
def predict_language(text):
    """
    Function which takes text as input and outputs the language

    Input: str
    Takes a text as input.

    Return: str
    Returns abbrevation of language.

    Example:
    >>> predict_language("Data Science in Action is cool!")
    'en'
    
    """

    prediction = pipe(text, top_k = 1, truncation = True)

    return prediction[0]['label']

# Apply function to 'text_content' column, update 'language'
data['language'] = data['text_content'].progress_apply(predict_language)

# Look at all Languages
lang = data['language'].value_counts()

# To simplify process - force all non-italian languages to english
data.loc[data['language'] != 'en', 'language'] = 'it'

# Copy dataset for emoji
data_no_emojis = data.copy()

### Processing emojis - Changing emojis to their word significant

def emoji2concat_description(text, lang='en'):
    """
    Function which takes a text, transforms emoji within text to their description,
    and concatenates them to the end of the text.
    
    Parameters:
    text (str): Text from which emojis are taken from.
    
    lang (str): Language of text. Default is set to English.
    
    Returns:
    text (str): Text with emojis transformed to their description and concatenated to end of text.

    Example:
    >>> emoji2concat_description(' 😂 That is so funny.', lang = 'en')
    'That is so funny. face with tears of joy'
    """

    # List of dictionaries of emojis in text
    emoji_list = emoji.emoji_list(text)

    # Strips emojis 
    ret = emoji.replace_emoji(text, replace='').strip()

    # Iterates through list
    for json in emoji_list:
    
        # Accesses emoji descriptions
        emoji_data = emoji.EMOJI_DATA[json['emoji']]

        # Gets description
        if isinstance(emoji_data, dict):

            emoji_desc = emoji_data.get(lang, emoji_data.get('en', ''))

        else:

            emoji_desc = emoji_data
        
        # Concatenate description
        this_desc = ' '.join(emoji_desc.split('_')).strip(':')

        ret += ' ' + this_desc

    return ret

# Iterate through data
for index, row in data.iterrows():

    # Acces language and text
    lang = row['language']

    text = row['text_content']

    # Update text
    new_text = emoji2concat_description(text, lang)

    # Update data
    data.at[index, 'text_content'] = new_text


def remove_emojis(text):
    """ 
    Removes emojis from text.
    
    Parameters:
    text (str): Takes text as input
    
    Returns:
    new_text (str): Text withour emojis

    Example:
    >>> remove_emojis(' 😂 That is so funny.')
    'That is so funny.'
    
    """

    new_text = emoji.replace_emoji(text, replace='')

    return new_text

# Remove emojis
data_no_emojis['text_content'] = data_no_emojis['text_content'].apply(remove_emojis)

### NLP Preprocessing Steps:
### 1 - Remove non-alphabetic characters 
### 2 - Convert uppercase to lowercase 
### 3 - Tokenize
### 4 - Remove stopwords
### 5 - Lemmatization

### 1 - Remove non-alphabetic characters 
data['text_content'] = data['text_content'].str.replace('[^a-zA-ZàèéìòùÀÈÉÌÒÙ ]', '', regex = True)
data_no_emojis['text_content'] = data_no_emojis['text_content'].str.replace('[^a-zA-ZàèéìòùÀÈÉÌÒÙ ]', '', regex = True)

# Check if any empty or whitespace only now
empty_or_whitespace = (data['text_content'] == '') | (data['text_content'].str.strip() == '')
empty_or_whitespace_no_emojis = (data_no_emojis['text_content'] == '') | (data_no_emojis['text_content'].str.strip() == '')

# Remove & reset index
data = data[~empty_or_whitespace]
data.reset_index(drop = True)

data_no_emojis = data_no_emojis[~empty_or_whitespace_no_emojis]
data_no_emojis.reset_index(drop = True)

### 2 - Convert uppercase to lowercase 
data['text_content'] = data['text_content'].str.lower()
data_no_emojis['text_content'] = data_no_emojis['text_content'].str.lower()

### 3 - Tokenize

# Run following line if need to download - commented out for now
# nltk.download('punkt')

data['tokenized'] = data['text_content'].apply(lambda x: word_tokenize(x) if isinstance(x, str) else [])
data_no_emojis['tokenized'] = data_no_emojis['text_content'].apply(lambda x: word_tokenize(x) if isinstance(x, str) else [])

### 4 - Remove stopwords 

# Run following line if need to download - commented out for now
#nltk.download('stopwords')

stop_words = set(stopwords.words('italian'))
en_stopwords = set(stopwords.words('english'))

stop_words.update(en_stopwords)

domain_words = ['chiara', 'ferragni', 'balocco', 'pandoro', 'italian', 'italiana','influencer']

stop_words.update(domain_words)

# Remove stopwords and domain words from text
data['filtered'] = data['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])
data_no_emojis['filtered'] = data_no_emojis['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])

### 5 - Stemming - Will Lemmatize Instead

# Run following line if need to download - commented out for now
# nltk.download('wordnet')
# nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Lemmatize text
data['lemmatized'] = data['filtered'].apply(lambda token_list: [lemmatizer.lemmatize(token) for token in token_list])
data_no_emojis['lemmatized'] = data_no_emojis['filtered'].apply(lambda token_list: [lemmatizer.lemmatize(token) for token in token_list])

### Splitting datasets into seperate languages

# Italian
it = data[data['language'] == 'it']
it_ne = data_no_emojis[data_no_emojis['language'] == 'it']

# English
en = data[data['language'] == 'en']
en_ne = data_no_emojis[data_no_emojis['language'] == 'en']

### Rejoin words together to use model's tokenizer
it['sentence'] = it['lemmatized'].apply(lambda tokens: ' '.join(tokens))
it_ne['sentence'] = it_ne['lemmatized'].apply(lambda tokens: ' '.join(tokens))

en['sentence'] = en['lemmatized'].apply(lambda tokens: ' '.join(tokens))
en_ne['sentence'] = en_ne['lemmatized'].apply(lambda tokens: ' '.join(tokens))

################ SENTIMENT ANALYSIS ################

### Italian Data First
### Sentiment analysis with: https://huggingface.co/neuraly/bert-base-italian-cased-sentiment?text=Huggingface+%C3%A8+un+team+fantastico%21

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("neuraly/bert-base-italian-cased-sentiment")

# Load the model, use .cuda() to load it on the GPU
model = AutoModelForSequenceClassification.from_pretrained("neuraly/bert-base-italian-cased-sentiment")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def italian_sentiment(sentence):
    """
    Returns sentiment of an inputted italian text.

    Parameters:
    sentence (str): Text sentence to be analyzed.

    Return:
    sentiment_score (dict): Dictionary of sentiment scores.

    Example:
    >>> italian_sentiment('Huggingface è un team fantastico!')
    {'negative': 0.000, 'neutral': 0.003, 'positive':0.997}
    """
    # Encode the sentence using the tokenizer
    input_ids = tokenizer.encode(sentence, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512)
    
    # Transfer tensor to the same device as model
    input_ids = input_ids.to(device)
    
    # Perform prediction
    with torch.no_grad():
        logits = model(input_ids).logits
    
    # Compute probabilities using softmax
    probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Define sentiment labels
    labels = ['negative', 'neutral', 'positive']
    
    # Create a dictionary of labels with corresponding probabilities
    sentiment_scores = {label: prob for label, prob in zip(labels, probabilities)}

    return sentiment_scores

# Apply functions
it['sentiment'] = it['sentence'].progress_apply(italian_sentiment)
it_ne['sentiment'] = it_ne['sentence'].progress_apply(italian_sentiment)

### Italian Emotions
### Done with: https://huggingface.co/MilaNLProc/feel-it-italian-emotion?text=Mi+piaci.+Ti+amo

# Load model - top_k is the number of predictions to return, truncation is used to seperate texts if over max_length of 512 token
it_emotion_classifier = pipeline("text-classification", model = 'e', top_k = 4, truncation = True, max_length = 512)

def italian_emotion(sentence):
    """
    Returns emotions of an inputted italian text.

    Parameters:
    sentence (str): Text sentence to be analyzed.

    Return:
    prediction (list): List of lists of dictionaries with emotion labels and respective scores

    Example:
    >>> italian_emotion('Mi piaci. Ti amo')
    [[{'label':'joy', 'score': 0.999}, {'label': 'sadness', 'score': 0.000}, {'label': 'fear', 'score': 0.000}, {'label': 'anger', 'score': 0.000}]]
    """

    prediction = it_emotion_classifier(sentence)

    return prediction

# Initialize columns
it['emotion'] = pd.NA
it_ne['emotion'] = pd.NA

# Apply function
it['emotion'] = it['sentence'].progress_apply(italian_emotion)
it_ne['emotion'] = it_ne['sentence'].progress_apply(italian_emotion)

### English Data Next

# Sentiment Analysis
# Done with: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

# Load model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

def english_sentiment(sentence):
    """
    Returns sentiment of an inputted english text.

    Parameters:
    sentence (str): Text sentence to be analyzed.

    Return:
    ordered_results (dict): Dictionary of sentiment scores.

    Example:
    >>> english_sentiment('Covid cases are increasing fast!')
    {'negative': 0.724, 'neutral': 0.229, 'positive':0.048}
    """
    # Tokenize text
    encoded_input = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)

    output = model(**encoded_input)

    scores = output[0][0].detach().numpy()

    # Uses softmax as activation
    scores = softmax(scores)

    results = {}

    for i in range(len(scores)):

        label = config.id2label[i]

        score = np.round(float(scores[i]), 4)

        results[label] = score

    # Order results
    ordered_results = {'negative': results['negative'], 'neutral': results['neutral'], 'positive': results['positive']}

    return ordered_results

# Apply function
en['sentiment'] = en['sentence'].progress_apply(english_sentiment)
en_ne['sentiment'] = en_ne['sentence'].progress_apply(english_sentiment)

# English Emotion
# Done with: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

en_emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k = None, truncation = True, max_length = 512)

def english_emotion(sentence):
    """
    Returns emotions of an inputted english text.

    Parameters:
    sentence (str): Text sentence to be analyzed.

    Return:
    prediction (list): list of list of dictionaries of emotions scores. Possible emotions are: sadness, neutral, surprise,
    joy, disgust, and anger.

    Example:
    >>> english_emotion('This movie always makes me cry..')
    [[{'label': 'sadness', 'score': 0.8657137751579285}, {'label': 'neutral', 'score': 0.046896398067474365}, {'label': 'surprise', 'score': 0.04181886836886406},
    {'label': 'disgust', 'score': 0.01544109731912613}, {'label': 'joy', 'score': 0.014266788959503174}, {'label': 'fear', 'score': 0.00932026281952858},
    {'label': 'anger', 'score': 0.006542800460010767}]]
    """

    prediction = en_emotion_classifier(sentence)

    return prediction

en['emotion'] = pd.NA
en_ne['emotion'] = pd.NA

# Apply function
en['emotion'] = en['sentence'].progress_apply(english_emotion)
en_ne['emotion'] = en_ne['sentence'].progress_apply(english_emotion)

### Remerge split data

final_data = pd.concat([it, en])
final_data_ne = pd.concat([it_ne, en_ne])

# Transform emotion column
def extract_and_transform(nested_list):
    """
    Function extracts dictionary from nested lists and transforms it.

    Parameters:
    nest_list (list): A nested list.

    Returns:
    result (dictionary): Modified dictionary extracted from nested list

    Example:
    >>> extract_and_transform([[{'label': 'anger', 'score': 0.6105449795722961}, {'label': 'sadness', 'score': 0.3431888520717621}, 
        {'label': 'disgust', 'score': 0.01543237641453743}, {'label': 'neutral', 'score': 0.013709764927625656}]])
    {'anger': 0.6105449795722961, 'sadness': 0.3431888520717621, 'disgust': 0.01543237641453743, 'neutral': 0.013709764927625656}
    """

    result = {}

    for list in nested_list:

        for dictionary in list:

            label = dictionary['label']

            score = dictionary['score']

            result[label] = score

    return result

# Apply function
final_data['emotion'] = final_data['emotion'].apply(extract_and_transform)
final_data_ne['emotion'] = final_data_ne['emotion'].apply(extract_and_transform)

################ COMPARING RESULTS ################

# Social Media

insta = final_data[final_data['source_category'] == 'Instagram']
X = final_data[final_data['source_category'] == 'X']
reddit = final_data[final_data['source_category'] == 'Reddit']
article = final_data[final_data['type_category'] == 'article']

# Social Media Without Emojis

insta_ne = final_data_ne[final_data_ne['source_category'] == 'Instagram']
X_ne = final_data_ne[final_data_ne['source_category'] == 'X']
reddit_ne = final_data_ne[final_data_ne['source_category'] == 'Reddit']
article_ne = final_data_ne[final_data_ne['type_category'] == 'article']

# Her Instagram & Her Brand - With and without emojis

cf_insta = final_data[final_data['post_creator'] == 'chiaraferragni']
cfb_insta = final_data[final_data['post_creator'] == 'chiaraferragnibrand']

cf_insta_ne = final_data_ne[final_data_ne['post_creator'] == 'chiaraferragni']
cfb_insta_ne = final_data_ne[final_data_ne['post_creator'] == 'chiaraferragnibrand']

# Insta Comments without her

insta_no_cf = insta[(insta['post_creator'] != 'chiaraferragni') & (insta['post_creator'] != 'chiaraferragnibrand')]
insta_no_cf_ne = insta_ne[(insta_ne['post_creator'] != 'chiaraferragni') & (insta_ne['post_creator'] != 'chiaraferragnibrand')]

# English & Italian
en = final_data[final_data['language'] == 'en']
en_ne = final_data_ne[final_data_ne['language'] == 'en']

it = final_data[final_data['language'] == 'it']
it_ne = final_data_ne[final_data_ne['language'] == 'it']

# Averages sentiment to display mean sentiment
def sentiment_mean(df, column_name='sentiment'):
    """
    Calculates mean of sentiments.
    
    Parameters:
    df (str): name of dataset.

    column_name (str): name of column. Default is 'sentiment'.

    Returns:
    averages (dict): dictionary with average of each sentiment.
    """

    # Initialize sums for each sentiment
    sums = {'negative': 0, 'neutral': 0, 'positive': 0}

    # Initialize count for each sentiment
    count = 0
    
    for sentiment_dict in df[column_name]:

        for key in sentiment_dict:
            sums[key] += sentiment_dict[key]

        count += 1
    
    # Calculate averages
    averages = {key: sums[key] / count for key in sums}
    
    return averages

# Counts which sentiment had the highest probability per item
def count_max_sentiment(df, column_name):
    """
    Calculates instances in which sentiment had the highest probability per item.
    
    Parameters:
    df (str): name of dataset.

    column_name (str): name of column.

    Returns:
    max_sentiment_count (dict): dictionary with count of each sentiment.
    """
    # Initialize counter
    max_sentiment_count = Counter()
    
    for sentiment_dict in df[column_name]:

        # Sentiment with max prob
        max_sentiment = max(sentiment_dict, key=sentiment_dict.get)

        max_sentiment_count[max_sentiment] += 1
    
    return dict(max_sentiment_count)

# Averages emotion to display mean emotion
def emotion_mean(df, column = 'emotion'):
    """
    Calculates mean of emotion
    
    Parameters:
    df (str): name of dataset.

    column_name (str): name of column. Default is 'emotion'.

    Returns:
    sorted_emotion_means (dict): dictionary with average of each emotion.
    """

    # Initialize dict
    emotion_sums = {}

    # Initialize count
    count = 0
    
    # Iterate through each row's emotion data
    for data in df[column]:

        count += 1

        for emotion, value in data.items():

            if emotion in emotion_sums:

                emotion_sums[emotion] += value

            else:

                emotion_sums[emotion] = value
    
    # Calculate the mean for emotions
    emotion_means = {emotion: total / count for emotion, total in emotion_sums.items()}
    
    sorted_emotion_means = dict(sorted(emotion_means.items(), key=lambda item: item[1], reverse=True))

    return sorted_emotion_means

# Counts which emotion had the highest probability per item
def count_max_emotion(df, column_name):
    """
    Calculates instances in which specific emotion had the highest probability per item.
    
    Parameters:
    df (str): name of dataset.

    column_name (str): name of column.

    Returns:
    max_emotion_count (dict): dictionary with count of each sentiment.
    """
    # Initialize counter
    max_emotion_count = Counter()
    
    for emotion_dict in df[column_name]:

        # Find the emotion with max score
        max_emotion = max(emotion_dict, key=emotion_dict.get)

        # Increment the counter for this sentiment
        max_emotion_count[max_emotion] += 1
    
    return dict(max_emotion_count)

# Data Source
data_sources = {
     'Total' : final_data,
     'Total w/o Emojis': final_data_ne,
     'Instagram': insta,
     'Instagram w/o Emojis': insta_ne,
     'Chiara Ferragni Instagram': cf_insta,
     'Chiara Ferragni Instagram w/o Emojis': cf_insta_ne,
     'Chiara Ferragni Brand Instagram': cfb_insta,
     'Chiara Ferragni Brand Instagram w/o Emojis': cfb_insta_ne,
     'Instagram w/o Chiara Ferragni Affiliated Posts': insta_no_cf,
     'Instagram w/o Chiara Ferragni Affiliated Post and w/o Emojis': insta_no_cf_ne,
     'X': X,
     'X w/o emojis': X_ne,
     'Reddit': reddit,
     'Reddit w/o Emojis': reddit_ne,
     'Articles': article,
     'Artciles w/o Emojis': article_ne,
     'Italian': it,
     'Italian w/o Emojis': it_ne,
     'English': en,
     'English w/o Emojis': en_ne
}

# Results DF
def process_data(df, column_name):
    """
    Applies function to a dataframe.

    Input:
    df (str): name of a dataframe

    Returns:
    results(pd.DataFrame): DataFrame with applied functions
    """

    results = {"Sentiment Mean": sentiment_mean(df, 'sentiment'),
            "Max Sentiment Count": count_max_sentiment(df, 'sentiment'),
            "Emotion Mean": emotion_mean(df, 'emotion'),
            "Max Emotion Count": count_max_emotion(df, 'emotion')}

    return results

# Created to optimize DataFrame extraction
def prepare_and_process_data():

    results = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:

        # Future map returns results in the order they were started
        futures = {executor.submit(process_data, df, name): name for name, df in data_sources.items()}

        for future in concurrent.futures.as_completed(futures):

            name = futures[future]
            try:
                results[name] = future.result()

            except Exception as e:

                print(f"Failed to process data for {name}: {str(e)}")

    return results

results = prepare_and_process_data()

# Convert results to DataFrame
results = pd.DataFrame.from_dict(results, orient='index').applymap(str)

# Order the results
results = results.reindex(data_sources.keys())

# Save to excel

final_data.to_csv('Final_Data.csv', index = False)
final_data_ne.to_csv('Final_Data_No_Emojis.csv', index = False)
results.to_csv('Results.csv', index = True)

final_data = pd.read_csv('/Users/davidpaquette/Downloads/Final_Data.csv')
final_data_ne = pd.read_csv('/Users/davidpaquette/Downloads/Final_Data_No_Emojis.csv')

################ VISUALIZATIONS AS A FUNCTION OF TIME ################

final_data['month_year'] = pd.to_datetime(final_data['date']).dt.to_period('M')
final_data_ne['month_year'] = pd.to_datetime(final_data_ne['date']).dt.to_period('M')

monthly_data = final_data[['source_category', 'post_creator','month_year', 'sentiment']].copy()
monthly_data_ne = final_data_ne[['source_category', 'post_creator','month_year', 'sentiment']].copy()

sentiments_expanded = monthly_data['sentiment'].apply(pd.Series)
sentiments_expanded_ne = monthly_data_ne['sentiment'].apply(pd.Series)

# Combine the expanded sentiments back into the original DataFrame
monthly_data = pd.concat([monthly_data.drop('sentiment', axis=1), sentiments_expanded], axis=1)
monthly_data_ne = pd.concat([monthly_data_ne.drop('sentiment', axis=1), sentiments_expanded_ne], axis=1)

clean_monthly_data = monthly_data.dropna(subset=['month_year'])
clean_monthly_data_ne = monthly_data_ne.dropna(subset=['month_year'])

# Group by 'month_year' and calculate the mean for each sentiment
monthly_sentiment_mean = clean_monthly_data.groupby('month_year')[['negative', 'neutral', 'positive']].mean().reset_index()
monthly_sentiment_mean_ne = clean_monthly_data_ne.groupby('month_year')[['negative', 'neutral', 'positive']].mean().reset_index()

# Emotions

monthly_emotions = final_data[['source_category', 'post_creator','month_year', 'emotion']].copy()
monthly_emotions_ne = final_data_ne[['source_category', 'post_creator','month_year', 'emotion']].copy()

emotions_expanded = monthly_emotions['emotion'].apply(pd.Series)
emotions_expanded_ne = monthly_emotions_ne['emotion'].apply(pd.Series)

# Combine the expanded sentiments back into the original DataFrame
monthly_emotions = pd.concat([monthly_emotions.drop('emotion', axis=1), emotions_expanded], axis=1)
monthly_emotions_ne = pd.concat([monthly_emotions_ne.drop('emotion', axis=1), emotions_expanded_ne], axis=1)

clean_monthly_emotions = monthly_emotions.dropna(subset=['month_year'])
clean_monthly_emotions_ne = monthly_emotions_ne.dropna(subset=['month_year'])

# Group by 'month_year' and calculate the mean for each sentiment
monthly_emotions_mean = clean_monthly_emotions.groupby('month_year')[['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']].mean().reset_index()
monthly_emotions_mean_ne = clean_monthly_emotions_ne.groupby('month_year')[['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']].mean().reset_index()

### Her Instagram

monthly_cf_sentiment = monthly_data[monthly_data['post_creator'] == 'chiaraferragni']
monthly_cf_sentiment_ne = monthly_data_ne[monthly_data_ne['post_creator'] == 'chiaraferragni']

cf_sentiments_expanded = monthly_cf_sentiment['sentiment'].apply(pd.Series)
cf_sentiments_expanded_ne = monthly_cf_sentiment_ne['sentiment'].apply(pd.Series)

cf_monthly_data = pd.concat([monthly_cf_sentiment.drop('sentiment', axis=1), cf_sentiments_expanded], axis=1)
cf_monthly_data_ne = pd.concat([monthly_cf_sentiment_ne.drop('sentiment', axis=1), cf_sentiments_expanded_ne], axis=1)

cf_clean_monthly_data = cf_monthly_data.dropna(subset=['month_year'])
cf_clean_monthly_data_ne = cf_monthly_data_ne.dropna(subset=['month_year'])

# Group by 'month_year' and calculate the mean for each sentiment
cf_monthly_sentiment_mean = cf_clean_monthly_data.groupby('month_year')[['negative', 'neutral', 'positive']].mean().reset_index()
cf_monthly_sentiment_mean_ne = cf_clean_monthly_data_ne.groupby('month_year')[['negative', 'neutral', 'positive']].mean().reset_index()

# Emotions

cf_emotions_expanded = monthly_cf_sentiment['emotion'].apply(pd.Series)
cf_emotions_expanded_ne = monthly_cf_sentiment_ne['emotion'].apply(pd.Series)

# Combine the expanded sentiments back into the original DataFrame
cf_monthly_emotions = pd.concat([monthly_cf_sentiment.drop('emotion', axis=1), cf_emotions_expanded], axis=1)
cf_monthly_emotions_ne = pd.concat([monthly_cf_sentiment_ne.drop('emotion', axis=1), cf_emotions_expanded_ne], axis=1)

cf_clean_monthly_emotions = cf_monthly_emotions.dropna(subset=['month_year'])
cf_clean_monthly_emotions_ne = cf_monthly_emotions_ne.dropna(subset=['month_year'])

# Group by 'month_year' and calculate the mean for each sentiment
cf_monthly_emotions_mean = cf_clean_monthly_emotions.groupby('month_year')[['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']].mean().reset_index()
cf_monthly_emotions_mean_ne = cf_clean_monthly_emotions_ne.groupby('month_year')[['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']].mean().reset_index()

### All of Instagram Except Her

monthly_no_cf_sentiment = final_data[(final_data['post_creator'] != 'chiaraferragni') & (final_data['post_creator'] != 'chiaraferragnibrand') & (final_data['source_category'] == 'Instagram')]
monthly_no_cf_sentiment_ne = final_data_ne[(final_data_ne['post_creator'] != 'chiaraferragni') & (final_data_ne['post_creator'] != 'chiaraferragnibrand') & (final_data_ne['source_category'] == 'Instagram')]

no_cf_sentiments_expanded = monthly_no_cf_sentiment['sentiment'].apply(pd.Series)
no_cf_sentiments_expanded_ne = monthly_no_cf_sentiment_ne['sentiment'].apply(pd.Series)

no_cf_monthly_data = pd.concat([monthly_no_cf_sentiment.drop('sentiment', axis=1), no_cf_sentiments_expanded], axis=1)
no_cf_monthly_data_ne = pd.concat([monthly_no_cf_sentiment_ne.drop('sentiment', axis=1), no_cf_sentiments_expanded_ne], axis=1)

no_cf_clean_monthly_data = no_cf_monthly_data.dropna(subset=['month_year'])
no_cf_clean_monthly_data_ne = no_cf_monthly_data_ne.dropna(subset=['month_year'])

# Group by 'month_year' and calculate the mean for each sentiment
no_cf_monthly_sentiment_mean = no_cf_clean_monthly_data.groupby('month_year')[['negative', 'neutral', 'positive']].mean().reset_index()
no_cf_monthly_sentiment_mean_ne = no_cf_clean_monthly_data_ne.groupby('month_year')[['negative', 'neutral', 'positive']].mean().reset_index()

# Emotions

no_cf_emotions_expanded = monthly_no_cf_sentiment['emotion'].apply(pd.Series)
no_cf_emotions_expanded_ne = monthly_no_cf_sentiment_ne['emotion'].apply(pd.Series)

# Combine the expanded sentiments back into the original DataFrame
no_cf_monthly_emotions = pd.concat([monthly_no_cf_sentiment.drop('emotion', axis=1), no_cf_emotions_expanded], axis=1)
no_cf_monthly_emotions_ne = pd.concat([monthly_no_cf_sentiment_ne.drop('emotion', axis=1), no_cf_emotions_expanded_ne], axis=1)

no_cf_clean_monthly_emotions = no_cf_monthly_emotions.dropna(subset=['month_year'])
no_cf_clean_monthly_emotions_ne = no_cf_monthly_emotions_ne.dropna(subset=['month_year'])

# Group by 'month_year' and calculate the mean for each sentiment
no_cf_monthly_emotions_mean = no_cf_clean_monthly_emotions.groupby('month_year')[['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']].mean().reset_index()
no_cf_monthly_emotions_mean_ne = no_cf_clean_monthly_emotions_ne.groupby('month_year')[['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']].mean().reset_index()

################ VISUALIZATIONS ################

# Events to compare to
events = [
    {"data": "2023-12-14", "description": "Discovery and Dissemination"},
    {"data": "2023-12-18", "description": "Apology Video"},
    {"data": "2023-12-21", "description": "Eyewear brand Safilo ends collaboration"},
    {"data": "2024-01-05", "description": "Coca-Cola ends collaboration"},
    {"data": "2024-01-09", "description": "Official accusations Legal Response"},
    {"data": "2024-01-24", "description": "Charge of aggravated fraud"},
    {"data": "2024-03-03", "description": "Appearance on 'Che tempo che fa'"},
    {"data": "2024-03-08", "description": "L'Espresso Joker Cover"}
]

events = pd.DataFrame(events)
events['data'] = pd.to_datetime(events['data'])

# Colors

color_map = {'negative': '#0af6ee', 'neutral': '#bbbbbb', 'positive': '#ff7bef'}

monthly_sentiment_mean['month_year'] = monthly_sentiment_mean['month_year'].dt.to_timestamp()
monthly_sentiment_mean_ne['month_year'] = monthly_sentiment_mean_ne['month_year'].dt.to_timestamp()

fig = px.line(monthly_sentiment_mean, x='month_year', y=['negative', 'neutral', 'positive'],
              labels={'value': 'Sentiment Score', 'month_year': 'Date'},
              title='Sentiment Score Over Time',
              line_shape='linear',
              color_discrete_map=color_map)

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })

    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })


fig.update_xaxes(range=['2023-11', monthly_sentiment_mean['month_year'].max().strftime('%Y-%m-%d')],
                 color='black',  
                 tickfont=dict(color='black'),  
                 title_font=dict(color='black') 
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'), 
    title_font=dict(color='black') 
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Sentiment',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black' 
)

fig.show()

# Show the 

fig = px.line(monthly_sentiment_mean_ne, x='month_year', y=['negative', 'neutral', 'positive'],
              labels={'value': 'Sentiment Levels', 'month_year': 'Date'},
              title='Sentiment Over Time W/o Emojis',
              line_shape = 'linear',
              color_discrete_map = color_map)

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })
    
    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })


fig.update_xaxes(range=['2023-11', monthly_sentiment_mean_ne['month_year'].max().strftime('%Y-%m-%d')],
                 color='black',  
                 tickfont=dict(color='black'), 
                 title_font=dict(color='black')  
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'),  
    title_font=dict(color='black')  
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Sentiment',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)

# Show figure
fig.show()


# Drop Columns for Visualizations
monthly_emotions_mean = monthly_emotions_mean.drop(columns=['disgust', 'neutral', 'surprise'])
monthly_emotions_mean_ne = monthly_emotions_mean_ne.drop(columns=['disgust', 'neutral', 'surprise'])

monthly_emotions_mean['month_year'] = monthly_emotions_mean['month_year'].dt.to_timestamp()
monthly_emotions_mean_ne['month_year'] = monthly_emotions_mean_ne['month_year'].dt.to_timestamp()

fig = px.line(monthly_emotions_mean, x='month_year', y=['anger', 'fear', 'joy', 'sadness'],
              labels={'value': 'Emotion Levels', 'month_year': 'Date'},
              title='Emotion Trends Over Time',
              line_shape = 'linear')

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })
    
    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0, 
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })


fig.update_xaxes(range=['2023-11', monthly_emotions_mean['month_year'].max().strftime('%Y-%m-%d')],
                 color='black', 
                 tickfont=dict(color='black'),  
                 title_font=dict(color='black') 
)

fig.update_yaxes(
    color='black',
    tickfont=dict(color='black'), 
    title_font=dict(color='black') 
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Emotions',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)


# Show figure
fig.show()

fig = px.line(monthly_emotions_mean_ne, x='month_year', y=['anger', 'fear', 'joy', 'sadness'],
              labels={'value': 'Emotion Levels', 'month_year': 'Date'},
              title='Emotion Trends Over Time W/o Emojis')

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })

    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })


fig.update_xaxes(range=['2023-11', monthly_emotions_mean_ne['month_year'].max().strftime('%Y-%m-%d')],
                 color='black', 
                 tickfont=dict(color='black'), 
                 title_font=dict(color='black') 
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'), 
    title_font=dict(color='black')  
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Emotions',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)

# Show figure
fig.show()

### Her Instagram

cf_monthly_sentiment_mean['month_year'] = cf_monthly_sentiment_mean['month_year'].dt.to_timestamp()
cf_monthly_sentiment_mean_ne['month_year'] = cf_monthly_sentiment_mean_ne['month_year'].dt.to_timestamp()

fig = px.line(cf_monthly_sentiment_mean, x='month_year', y=['negative', 'neutral', 'positive'],
              labels={'value': 'Sentiment Levels', 'month_year': 'Date'},
              title='Sentiment on her Instagram Over Time',
              line_shape = 'linear',
              color_discrete_map = color_map)

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })

    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })


fig.update_xaxes(range=['2023-12', cf_monthly_sentiment_mean['month_year'].max().strftime('%Y-%m-%d')],
                 color='black',  
                 tickfont=dict(color='black'),  
                 title_font=dict(color='black')  
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'),  
    title_font=dict(color='black')  
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Sentiment',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)

# Show figure
fig.show()


fig = px.line(cf_monthly_sentiment_mean_ne, x='month_year', y=['negative', 'neutral', 'positive'],
              labels={'value': 'Sentiment Levels', 'month_year': 'Date'},
              title='Sentiment on her Instagram Over Time W/o Emojis',
              line_shape = 'linear',
              color_discrete_map = color_map)

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })

    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })


fig.update_xaxes(range=['2023-12', cf_monthly_sentiment_mean_ne['month_year'].max().strftime('%Y-%m-%d')],
                 color='black',  
                 tickfont=dict(color='black'),
                 title_font=dict(color='black')  
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'),  
    title_font=dict(color='black') 
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Sentiment',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)

# Show figure
fig.show()


cf_monthly_emotions_mean = cf_monthly_emotions_mean.drop(columns=['disgust', 'neutral', 'surprise'])
cf_monthly_emotions_mean_ne = cf_monthly_emotions_mean_ne.drop(columns=['disgust', 'neutral', 'surprise'])

cf_monthly_emotions_mean['month_year'] = cf_monthly_emotions_mean['month_year'].dt.to_timestamp()
cf_monthly_emotions_mean_ne['month_year'] = cf_monthly_emotions_mean_ne['month_year'].dt.to_timestamp()

fig = px.line(cf_monthly_emotions_mean, x='month_year', y=['anger', 'fear', 'joy', 'sadness'],
              labels={'value': 'Emotion Levels', 'month_year': 'Date'},
              title='Emotion Trends on her Instagram Over Time')

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })
   
    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })


fig.update_xaxes(range=['2023-12', cf_monthly_emotions_mean['month_year'].max().strftime('%Y-%m-%d')],
                 color='black',  
                 tickfont=dict(color='black'), 
                 title_font=dict(color='black')
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'), 
    title_font=dict(color='black') 
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Emotions',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)


# Show figure
fig.show()

fig = px.line(cf_monthly_emotions_mean_ne, x='month_year', y=['anger', 'fear', 'joy', 'sadness'],
              labels={'value': 'Emotion Levels', 'month_year': 'Date'},
              title='Emotion Trends on her Instagram Over Time W/o Emojis')

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })

    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })

# Update axes colors to black
fig.update_xaxes(range=['2023-12', cf_monthly_emotions_mean_ne['month_year'].max().strftime('%Y-%m-%d')],
                 color='black', 
                 tickfont=dict(color='black'),  
                 title_font=dict(color='black')  
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'),  
    title_font=dict(color='black')  
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Emotions',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)

# Show the figure
fig.show()

### All of Instagram Except Her

no_cf_monthly_sentiment_mean['month_year'] = no_cf_monthly_sentiment_mean['month_year'].dt.to_timestamp()
no_cf_monthly_sentiment_mean_ne['month_year'] = no_cf_monthly_sentiment_mean_ne['month_year'].dt.to_timestamp()

fig = px.line(no_cf_monthly_sentiment_mean, x='month_year', y=['negative', 'neutral', 'positive'],
              labels={'value': 'Sentiment Levels', 'month_year': 'Date'},
              title='Sentiment Over Time Excluding All Instagram Posts Affiliated with Her',
              line_shape = 'linear',
              color_discrete_map = color_map)

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })
    
    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })

# Update axes colors to black
fig.update_xaxes(range=['2023-12', no_cf_monthly_sentiment_mean['month_year'].max().strftime('%Y-%m-%d')],
                 color='black',  
                 tickfont=dict(color='black'),  
                 title_font=dict(color='black')  
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'),  
    title_font=dict(color='black')  
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Sentiment',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)

# Show the figure
fig.show()

fig = px.line(no_cf_monthly_sentiment_mean_ne, x='month_year', y=['negative', 'neutral', 'positive'],
              labels={'value': 'Sentiment Levels', 'month_year': 'Date'},
              title='Sentiment Over Time Excluding All Instagram Posts Affiliated with Her & W/o Emojis',
                            line_shape = 'linear',
              color_discrete_map = color_map)

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })
    
    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })

# Update axes colors to black
fig.update_xaxes(range=['2023-12', no_cf_monthly_sentiment_mean_ne['month_year'].max().strftime('%Y-%m-%d')],
                 color='black',  
                 tickfont=dict(color='black'),  
                 title_font=dict(color='black')  
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'),  
    title_font=dict(color='black')  
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Sentiment',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)

# Show the figure
fig.show()

no_cf_monthly_emotions_mean = no_cf_monthly_emotions_mean.drop(columns=['disgust', 'neutral', 'surprise'])
no_cf_monthly_emotions_mean_ne = no_cf_monthly_emotions_mean_ne.drop(columns=['disgust', 'neutral', 'surprise'])

no_cf_monthly_emotions_mean['month_year'] = no_cf_monthly_emotions_mean['month_year'].dt.to_timestamp()
no_cf_monthly_emotions_mean_ne['month_year'] = no_cf_monthly_emotions_mean_ne['month_year'].dt.to_timestamp()

fig = px.line(no_cf_monthly_emotions_mean, x='month_year', y=['anger', 'fear', 'joy', 'sadness'],
              labels={'value': 'Emotion Levels', 'month_year': 'Date'},
              title='Emotion Trends Over Time Excluding All Instagram Posts Affiliated with Her')

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })
    
    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })

# Update axes colors to black
fig.update_xaxes(range=['2023-11', no_cf_monthly_emotions_mean['month_year'].max().strftime('%Y-%m-%d')],
                 color='black',  
                 tickfont=dict(color='black'),  
                 title_font=dict(color='black')  
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'),  
    title_font=dict(color='black')  
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Emotions',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)

# Show the figure
fig.show()

fig = px.line(no_cf_monthly_emotions_mean_ne, x='month_year', y=['anger', 'fear', 'joy', 'sadness'],
              labels={'value': 'Emotion Levels', 'month_year': 'Date'},
              title='Emotion Trends Over Time Excluding All Instagram Posts Affiliated with Her W/o Emojis')

# Increase the line width
for trace in fig.data:
    trace.line.width = 4

# Adding shapes and annotations for events
shapes = []
annotations = []
for _, event in events.iterrows():
    date = pd.to_datetime(event['data'])
    shapes.append({
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': date,
        'y0': 0,
        'x1': date,
        'y1': 1,
        'line': {'color': 'green', 'width': 2},
    })
    
    annotations.append({
        'x': date,
        'y': 0.95,
        'xref': 'x',
        'yref': 'paper',
        'text': event['description'],
        'showarrow': True,
        'arrowhead': 7,
        'ax': 0,
        'ay': 0,  
        'bgcolor': 'white',
        'bordercolor': 'black',
        'borderwidth': 1,
        'borderpad': 2,
        'textangle': -90,
        'font': {'color': 'green', 'size': 14}
    })

# Update axes colors to black
fig.update_xaxes(range=['2023-11', no_cf_monthly_emotions_mean_ne['month_year'].max().strftime('%Y-%m-%d')],
                 color='black',  
                 tickfont=dict(color='black'),  
                 title_font=dict(color='black')  
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='black'),  
    title_font=dict(color='black')  
)

# Apply layout updates
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text='Emotions',
    xaxis_color = 'black',
    yaxis_color = 'black',
    plot_bgcolor='rgba(0,0,0,0)',  
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_color='black'  
)

# Show the figure
fig.show()
    
lang = pd.DataFrame(lang)
lang

fig = px.pie(lang, values='count', names=lang.index, title='Language Distribution', hole=0.4)  # The hole parameter creates the donut shape

# Display the name of the language next to the percentage on the pie chart
fig.update_traces(textinfo='percent+label')

# Hide the legend
fig.update_layout(showlegend=False)

# Set the background to be transparent
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

# Show the pie chart
fig.show()

### EXPORT TO CSV
monthly_sentiment_mean.to_csv('monthly_sentiment_mean.csv')
monthly_sentiment_mean_ne.to_csv('monthly_sentiment_mean_ne.csv')
monthly_emotions_mean.to_csv('monthly_emotions_mean.csv')
monthly_emotions_mean_ne.to_csv('monthly_emotions_mean_ne.csv')
cf_monthly_sentiment_mean.to_csv('cf_monthly_sentiment_mean.csv')
cf_monthly_sentiment_mean_ne.to_csv('cf_monthly_sentiment_mean_ne.csv')
cf_monthly_emotions_mean.to_csv('cf_monthly_emotions_mean.csv')
cf_monthly_emotions_mean_ne.to_csv('cf_monthly_emotions_mean_ne.csv')
no_cf_monthly_sentiment_mean.to_csv('no_cf_monthly_sentiment_mean.csv')
no_cf_monthly_sentiment_mean_ne.to_csv('no_cf_monthly_sentiment_mean_ne.csv')
no_cf_monthly_emotions_mean.to_csv('no_cf_monthly_emotions_mean.csv')
no_cf_monthly_emotions_mean_ne.to_csv('no_cf_monthly_emotions_mean_ne.csv')


max_emotion_count = {'anger': 12777, 'sadness': 1692, 'joy': 5817, 'fear': 1753, 'disgust': 16, 'surprise': 29, 'neutral': 170}
df = pd.DataFrame(list(max_emotion_count.items()), columns=['Emotion', 'Count'])

fig = px.bar(df, x='Emotion', y='Count',
             color='Emotion',  # Color bars by emotion
             color_discrete_map={
                 "anger": "#e76f51", "sadness": "#264653", "joy": "#f4a261",
                 "fear": "#2a9d8f", "disgust": "#8d99ae", "surprise": "#e9c46a",
                 "neutral": "#f2cc8f"  # Assigning custom colors
             })

# Set the background to be transparent
fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',  
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',  
    'title_font_color': 'green',  
    'font_color': 'green', 
})

# Update axes colors and labels to black and green
fig.update_xaxes(
    color='black', 
    tickfont=dict(color='green'),  
    title_font=dict(color='green') 
)

fig.update_yaxes(
    color='black',  
    tickfont=dict(color='green'),  
    title_font=dict(color='green') 
)

# Show the plot
fig.show()