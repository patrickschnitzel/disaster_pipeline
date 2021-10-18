import sys
import pickle
import re
from typing import List
from typing import Tuple
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath: str) -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """
    Loads data from SQL database into DataFrame

    Parameters
    ----------
    database_filepath
        path to sql database
    Returns
    -------
        X
            features
        Y
            labels
        y.names
            list of categorical column names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('DisasterResponseData',con=engine)
    X = df["message"]
    y = df.iloc[:,4:]
    return X, y, y.columns


def tokenize(text: str) -> List[str]:
    """
    tokenizes text
    
    Parameters
    ----------
    text
        text to tokenize
    Returns
    -------
        list of tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model() -> GridSearchCV:
    """
    builds the model. best model was extensively researched in the ML Pipeline Preparation notebook.
    
    Returns
    -------
        model
    """
    pipeline_linear_svc = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', OneVsRestClassifier(LinearSVC(random_state=0)))])
    parameters = {
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__ngram_range': ((1, 1), (1,2)),
        #'vect__max_features': (None, 5000,10000),
        #'tfidf__use_idf': (True, False)
    }
    return GridSearchCV(pipeline_linear_svc, param_grid=parameters, scoring="f1_micro" )


def evaluate_model(model: GridSearchCV, X_test: pd.Series, Y_test: pd.DataFrame, category_names: List[str]):
    """
    prints the classification report including relevant model metrics
    
    Parameters
    ----------
    model
        model to evaluate
    X_test
        test data
    Y_test
        test labels
    category_names
        names of categories
    """
    y_pred_test = model.predict(X_test)
    print(classification_report(Y_test, y_pred_test,target_names = category_names))


def save_model(model: GridSearchCV, model_filepath: str):
    """
    saves model to a pickle file
    
    Parameters
    ----------
    model
        model to save
    model_filepath
        path to save model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()