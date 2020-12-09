# %% md

import datetime
import re
from collections import Counter
from statistics import mean
from typing import List, NoReturn

import emojis
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import textstat
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer
from pandas import DataFrame
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# %%
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS

import os
from pathlib import Path


debug_dir = Path('debug')
if not os.path.exists(debug_dir.absolute()):
    os.makedirs(debug_dir)

NOT_NICE_TWEET_CODING = 'unpleasant'
NICE_TWEET_CODING = 'nice'

NORMAL_FLAG = 'normal'
ABUSIVE_FLAG = 'abusive'
HATEFUL_FLAG = 'hateful'

ALL_NORMALIZE_SETTINGS = ['pred', 'true', 'all']

#%% Begin section to mess with settings
WRITE_DEBUG_FILES = True

# Begin feature selection settings
SELECTIVELY_SAMPLE_TEST_SLICE = False # If set to true, we will attempt to adjust to have a less severe class imbalance in the test data.
tfidf_text_colmun_name = 'cleaned_no_flags'
#tfidf_text_colmun_name = 'cleaned_tweet'

USE_VOCAB_TFIDF = True
USE_TFIDF_EMOJI = True
USE_POS_TAG = True
USE_CHAR_TFIDF = True

# result_column = 'tweet_coding' # use 3 class
result_column = 'binary_class' # use 2 class

if result_column == 'tweet_coding':
    LABEL_LIST = [NORMAL_FLAG, ABUSIVE_FLAG, HATEFUL_FLAG]
elif result_column == 'binary_class':
    LABEL_LIST = [NICE_TWEET_CODING, NOT_NICE_TWEET_CODING]
# %%

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


print(datetime.datetime.now())

# %%

og_df = pd.read_csv('outfile.csv', usecols=['user_handle', 'tweet_text', 'tweet_coding'])
print(len(og_df))
print(og_df['tweet_coding'].value_counts())
og_df.drop(og_df[og_df['tweet_coding'] == 'spam'].index, inplace=True)
og_df.drop(og_df[og_df['tweet_coding'] == ''].index, inplace=True)

# df_to_annotate.dropna(axis=0, inplace=True)
print(og_df['tweet_coding'].value_counts())
print(datetime.datetime.today())

# %%

c = Counter(og_df['tweet_coding'])
[(type(k), k, c[k]) for k in c.keys()]
print(datetime.datetime.today())

# %%


obj = TweetTokenizer()
tmp_df = og_df.head(100)


def annotate_col_parts_of_speech(df_to_annotate: DataFrame, input_col: str = tfidf_text_colmun_name) -> NoReturn:
    input_text_series = df_to_annotate[input_col]

    def make_pos_sentence(tweet_text: str) -> str:
        tks = obj.tokenize(tweet_text)
        pos_list = [tag[1] for tag in pos_tag(tks)]
        return ' '.join(pos_list)

    df_to_annotate['pos_sentence'] = input_text_series.apply(make_pos_sentence)
    in_text_count = input_text_series.count()
    out_pos_count = df_to_annotate['pos_sentence'].count()
    assert in_text_count == out_pos_count, f'In- text {in_text_count} != {out_pos_count}'
    print(f'annotated {len(df_to_annotate)} rows')


# %%


sent_anlr = VS()


def balance_df_slice(df: DataFrame) -> DataFrame:
    abusive = df[df['tweet_coding'] == ABUSIVE_FLAG].sample(frac=.9)
    hateful = df[df['tweet_coding'] == 'hateful'].sample(frac=.9)
    normals_to_grab = int(mean([len(abusive), len(hateful)]))
    normal = df[df['tweet_coding'] == NORMAL_FLAG].head(normals_to_grab)
    return pd.concat([abusive, hateful, normal]).reindex()


train_df_slice = balance_df_slice(og_df)


mention_flag = 'USER_MENTION'
url_flag = 'URLHERE'
em_beg = u'\U0001f300'
em_end = u'\U0001fad6'
emoji_patt = re.compile(f'[{em_beg}-{em_end}]')

hashtag_patt = re.compile('#[a-zA-Z0-9]+')
pattern_replace_with_single_space = re.compile(f'(:|[\.]{3}|-|\s|{emoji_patt.pattern})+')
giant_url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_regex = re.compile('@[\w\-]+')
replace_w_empty_string_patt = re.compile("[']")
''' taken from https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/classifier/classifier.py#L42'''


def clean_tweet_text(text_string: str) -> str:
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) emoji with empty STring

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """

    parsed_text = giant_url_regex.sub(url_flag, text_string)
    parsed_text = mention_regex.sub(mention_flag, parsed_text)

    parsed_text = pattern_replace_with_single_space.sub(' ', parsed_text)
    parsed_text = emojis.decode(parsed_text)

    parsed_text = hashtag_patt.sub('HASHTAG', parsed_text)
    parsed_text = replace_w_empty_string_patt.sub('', parsed_text)
    return parsed_text.lower()


def get_num_mentioning(tweet_text: str) -> int:
    return tweet_text.count(mention_flag)


def get_num_urls(tweet_text: str) -> int:
    return tweet_text.count(url_flag)


def remove_flags(tweet_text: str) -> str:
    return tweet_text.replace(mention_flag, '').replace(url_flag, '')


stmr = PorterStemmer()
lmr = WordNetLemmatizer()


def stem_tweet(t: str) -> str:
    pieces = t.split(' ')
    l: List[str] = []
    for s in pieces:
        l.append(lmr.lemmatize(s))
    return ' '.join(l)


def get_sentiment_compound(t) -> float:
    return sent_anlr.polarity_scores(t)['compound']


# %%


def compute_tweet_features(df_to_clean: DataFrame):
    """
    Computes cleaned text, num mentioning, num urls, stemmed version of tweet, sentiment score, flesch score, and pos for each tweet
    :param df_to_clean: Pandas DataFrame
    :param remove_irrelevant: Whether or not to drop user_handle, tweet_creation, and follower_count
    :return:
    """
    tweet_series = df_to_clean['tweet_text']
    df_to_clean['cleaned_tweet'] = tweet_series.map(clean_tweet_text)
    cleaned_tweets = df_to_clean['cleaned_tweet']
    tweet_coding_series = df_to_clean['tweet_coding']

    df_to_clean['num_mentioning'] = cleaned_tweets.map(get_num_mentioning)
    df_to_clean['num_urls'] = cleaned_tweets.map(get_num_urls)

    df_to_clean['cleaned_no_flags'] = cleaned_tweets.map(remove_flags).map(stem_tweet)
    df_to_clean['binary_class'] = tweet_coding_series.map(
        lambda label: NICE_TWEET_CODING if label == NORMAL_FLAG else NOT_NICE_TWEET_CODING)

    df_to_clean['sentiment_score'] = cleaned_tweets.map(get_sentiment_compound)
    df_to_clean['flesch_score'] = tweet_series.map(textstat.flesch_reading_ease)
    df_to_clean['emojis'] = tweet_series.map(lambda t: ' '.join(emojis.iter(t)) or ' ')

    # Tag with parts of speech
    annotate_col_parts_of_speech(df_to_clean)
    return df_to_clean


train_df_slice = compute_tweet_features(train_df_slice)

train_df_slice[['tweet_text', 'cleaned_tweet', 'cleaned_no_flags']].to_csv(debug_dir.joinpath( 'trained_cleaned.csv'))
#%%
# Grab use those items not in the train slice to test
if SELECTIVELY_SAMPLE_TEST_SLICE:
    test_slice = balance_df_slice(og_df.drop(train_df_slice.index))
else:
    test_slice = og_df.drop(train_df_slice.index)
print(len(test_slice))

print(f'annotating og df_to_annotate this may take a moment len {len(og_df)}')
annotated_og_df = compute_tweet_features(og_df)
print('finished annotating og df_to_annotate')
print('annotating test df_to_annotate')
test_df_slice = compute_tweet_features(test_slice)
print('finished annotating test df_to_annotate')
print(datetime.datetime.today())

# %% md

# Build model

# %%


print('Training data stats')
print(train_df_slice[result_column].value_counts())
print(len(train_df_slice))

word_vect = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 3),
    lowercase=True,
    analyzer='word',
    norm='l2',
    sublinear_tf=True,
    stop_words='english')

char_vect = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 4), # Accordin to 'Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter' 1-4 works best
    lowercase=True,
    analyzer='char',
    norm='l2',
    sublinear_tf=True)

emoji_vectorizer = CountVectorizer(
    max_features=200,
    ngram_range=(1, 1),
    lowercase=False,
    analyzer='word',
    vocabulary=set(emojis.db.get_emoji_aliases().values())
)
pos_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    use_idf=False,
    smooth_idf=False,
    max_features=2000,
    lowercase=False
)


# %%


def get_features(tf_idf_word_vectorizer, tf_idf_char_vectorizer, pos_sent_vectorizer, emoji_vectorizer, df) -> np.array:
    feature_vectors = []
    other_features = np.array(
        [df['num_mentioning'].to_numpy(), df['flesch_score'].to_numpy(), df['sentiment_score'].to_numpy()]).transpose()
    if USE_VOCAB_TFIDF:
        tfidf_vocab_vec = tf_idf_word_vectorizer.transform(df[tfidf_text_colmun_name])
        feature_vectors.append(tfidf_vocab_vec)
    if USE_CHAR_TFIDF:
        tfidf_char_vec = tf_idf_char_vectorizer.transform(df[tfidf_text_colmun_name])
        feature_vectors.append(tfidf_char_vec)
    if USE_POS_TAG:
        tfidf_pos_vec = pos_sent_vectorizer.transform(df['pos_sentence'])
        feature_vectors.append(tfidf_pos_vec)

    if USE_TFIDF_EMOJI:
        tfidf_emoji = emoji_vectorizer.transform(df['emojis'])
        feature_vectors.append(tfidf_emoji)

    stacked = hstack(feature_vectors)
    return stacked


word_vect.fit(annotated_og_df[tfidf_text_colmun_name])
char_vect.fit(annotated_og_df[tfidf_text_colmun_name])

pos_vectorizer.fit(annotated_og_df['pos_sentence'])
emoji_vectorizer.fit(og_df['emojis'])

train_x_features = get_features(tf_idf_word_vectorizer=word_vect, tf_idf_char_vectorizer=char_vect, pos_sent_vectorizer=pos_vectorizer, emoji_vectorizer= emoji_vectorizer, df= train_df_slice)
#%%
train_y = train_df_slice[result_column]

# %%

models_and_tags = []

models_and_tags.append(
    (LogisticRegression(C=20, multi_class='ovr', max_iter=150), 'base_lr_with_char'),
)
models_and_tags.append(
    (LinearSVC(max_iter=10000, multi_class='ovr', loss='hinge', C=50), 'hinge_i_10000_with_char'),
)
models_and_tags.append(
    (LinearSVC(max_iter=2000, multi_class='ovr', loss='hinge', C=50), 'hinge_i_2000_with_char'),
)
# models_and_tags.append(LinearSVC(max_iter=10000, multi_class='crammer_singer'))


print('training model(s)')
for model, _ in models_and_tags:
    model.fit(train_x_features, train_y)
print('model(s) trained')

# %%


def make_plot_file_name(data_mode, tag, normalized_to) -> Path:
    smote_desc = get_smote_status()
    dir_path = f'./plots/{data_mode}_{smote_desc}_norm_to_{normalized_to}/'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return Path(f'{dir_path}plot{mdl_name}_{result_column}_{tfidf_text_colmun_name}_{tag}.png')


def get_smote_status():
    smote_desc = 'smoted' if SELECTIVELY_SAMPLE_TEST_SLICE else 'unsmoted'
    return smote_desc


def make_classification_report_file_path(data_mode, tag) -> Path:
    dir_path = f'./classif_reports/{data_mode}_{get_smote_status()}/'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return Path(f'{dir_path}report{mdl_name}_{result_column}_{tfidf_text_colmun_name}_{tag}.txt')


# %%
# Evaluate model

test_features_x = get_features(tf_idf_word_vectorizer=word_vect, tf_idf_char_vectorizer=char_vect, pos_sent_vectorizer=pos_vectorizer, emoji_vectorizer=emoji_vectorizer, df=test_slice)

y_test = test_slice[result_column]
figure_file = open('fig_list.md', 'w')

for model, tag in models_and_tags:

    y_predicted = model.predict(test_features_x)

    test_slice['predicted'] = y_predicted

    print(f'train dataset is len {len(train_df_slice)}')
    print(f'test dataset is len {len(test_slice)}')

    print('y test stats')
    print(y_test.value_counts())
    print(f'this is how we do against our test data {len(y_test)}')
    print(Counter(y_test))
    plot_confusion_matrix(model, X=test_features_x, y_true=y_test)

    if WRITE_DEBUG_FILES:
        mis_cat = test_slice.query('predicted != @result_column')
        print('mis cat')
        cols = ['cleaned_tweet', result_column, 'predicted', 'flesch_score', 'sentiment_score', 'tweet_text']
        rel_cols = mis_cat[cols]
        # print(rel_cols)

        rel_cols.to_csv(debug_dir.joinpath('mis_cat.csv'))
        mis_cat.query('predicted == @ABUSIVE_FLAG & @result_column == @NORMAL_FLAG').to_csv(debug_dir.joinpath( 'p_abuse_a_normal.csv'))
        mis_cat.query('predicted != @NORMAL_FLAG & @result_column == @NORMAL_FLAG').to_csv(debug_dir.joinpath( 'a_normal_p_-.csv'))
        mis_cat.query('predicted != @HATEFUL_FLAG & @result_column == @HATEFUL_FLAG')[cols].to_csv(debug_dir.joinpath( 'a_hateful_miscat.csv'))


    mdl_name = model.__class__.__name__
    print('-' * 16)
    print(mdl_name, tag)
    for normalize_setting in ALL_NORMALIZE_SETTINGS:
        conf_matrix_plt = plot_confusion_matrix(estimator=model, X=test_features_x, y_true=y_test,
                                                normalize=normalize_setting, labels=LABEL_LIST)
        plt.title(f'Confusion matrix {mdl_name} {normalize_setting}_test data')
        plot_file_name: Path = make_plot_file_name('test', tag, normalize_setting)
        plot_full_path = plot_file_name.absolute().__str__()
        figure_file.write(f'![{plot_full_path}]({plot_full_path})')

        plt.savefig(plot_file_name.absolute())
        plt.show()

    classif_report = classification_report(y_true=y_test, y_pred=y_predicted, labels=LABEL_LIST)
    print(classif_report)
    from sklearn.metrics import precision_recall_fscore_support
    print(precision_recall_fscore_support(y_true=y_test, y_pred=y_predicted, labels=LABEL_LIST))

    class_report_file_path = make_classification_report_file_path('test', tag).absolute().__str__()
    with open(class_report_file_path, 'w') as f_writes:
        f_writes.write(classif_report)
    figure_file.write(classif_report)
    figure_file.write('\n\n')
figure_file.close()
print(datetime.datetime.now())

# %%
for model, tag in models_and_tags:
    print('this is how we do against the training data, use to decide if we have overfit')
    print(train_y.value_counts())
    for normalize_setting in ALL_NORMALIZE_SETTINGS:
        conf_matrix_plt = plot_confusion_matrix(model, X=train_x_features, y_true=train_y, normalize=normalize_setting)
        mdl_name = model.__class__.__name__
        plt.title(f'Confusion matrix {mdl_name} {normalize_setting} train data')
        plt.savefig(make_plot_file_name('train', tag, normalized_to=normalize_setting))

        plt.show()
    print(classification_report(y_true=train_y, y_pred=model.predict(train_x_features)))
    print(datetime.datetime.today())
