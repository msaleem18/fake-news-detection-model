
## -------------------------------------
### LIBRARIERS NEEDED
## -------------------------------------
import pandas as pd
import numpy as np
import re
import os
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score
#from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras 
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nlp = spacy.load('en_core_web_sm')
import nltk
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

## -------------------------------------
## ----- READ FILES
## -------------------------------------
def create_df(train_path, test_path, label_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    test_labels = pd.read_csv(label_path)

    test_final = pd.merge(test_data, test_labels, how = "inner", left_on='id',right_on='id')

    train_data["wc_heading"] = train_data["title"].apply(lambda n: len(str(n).split()))
    train_data["wc_news"] = train_data["text"].apply(lambda n: len(str(n).split()))
    test_final["wc_heading"] = test_final["title"].apply(lambda n: len(str(n).split()))
    test_final["wc_news"] = test_final["text"].apply(lambda n: len(str(n).split()))

    return train_data, test_final


## -------------------------------------
## ----- PREPROCESS
## -------------------------------------

####################################################################################
# CLEAN
####################################################################################
def clean_text(each_text):

    each_text = str(each_text).lower()
    each_text = re.sub(r"what's", "what is ", each_text)
    each_text = re.sub(r"sheis", "she is ", each_text)
    
    each_text = re.sub(r"http\S+", "", each_text)
    each_text = re.sub(r"cannot", "can not", each_text) ##--different
    each_text = re.sub(r"whatis", "what is", each_text) ##--different
    each_text = re.sub(r"\'s", " is", each_text)
    #each_text = each_text.replace(r".", "", each_text)
    each_text = re.sub(r"\’s", " is", each_text)
    each_text = re.sub(r"\'ve", " have", each_text)
    each_text = re.sub(r"n\’t", " not", each_text)
    each_text = re.sub(r"n\'t", " not", each_text)
    each_text = re.sub(r"i\'m", "i am", each_text)
    each_text = re.sub(r"i\’m", "i am", each_text)
    each_text = re.sub(r"\'ve", " have", each_text)
    each_text = re.sub(r"\'re", " are ", each_text)
    each_text = re.sub(r"\’re", " are ", each_text)
    each_text = re.sub(r"\'d", " would ", each_text)
    each_text = re.sub(r"\'ll", " will ", each_text)
    each_text = re.sub(r"\’ll", " will ", each_text)
    each_text = re.sub(r"\\", "", each_text)
    each_text = re.sub(r"\'", "", each_text)
    each_text = re.sub(r"\"", "", each_text)
    each_text = re.sub(r" ;",";", each_text)
    #each_text = re.sub('[^a-zA-Z; ?!]+', '', each_text)
    #each_text = re.sub(r'\d+', '', each_text)
    each_text = re.sub(r'&amp;?', r'and', each_text)  # replace & -> and
    each_text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", each_text)  # Remove URLs
    each_text = re.sub('[^A-Za-z0-9; ]+', '', each_text) ##keep only words and digits
    each_text = re.sub('\s+', ' ', each_text).strip()  # Remove and double spaces

    each_text = each_text.strip()
    
        
    return " ".join(each_text.split())

def clean_text_dataframe(df, col_to_use, stop_words_nltk):
    df[col_to_use+'_step_clean'] = df[col_to_use].apply(lambda x: clean_text(x))
    list_of_all_text = df[col_to_use+'_step_clean'].values.tolist()
    #print("type of list: %s" %type(list_of_all_text))
    #print(list_of_all_text)

    docs = list(nlp.pipe(list_of_all_text))
    docs_clean = []
    for single_doc in docs:
        #print(single_doc)
        single_doc_tokens = []
        for token in single_doc:
            #print(token.text, token.lemma_, token.pos_, token.is_digit, token.is_alpha)
            if (token.is_alpha or token.pos_ == 'NUM') and (token.lemma_ not in stop_words_nltk):
                #print("in here")
                #print(word_count_dict[token.lemma_])
                #if token.lemma_ in word_count_dict:
                 #   print(word_count_dict[token.lemma_])
                    #if word_count_dict[token.lemma_] > 1:
                single_doc_tokens.append(token.lemma_)
                #else:
                 #  single_doc_tokens.append(token.lemma_)

        docs_clean.append(" ".join(single_doc_tokens))
    
    return docs_clean

def start_df_cleaning(df):
    # stop words
    ###-----
    nltk.download('stopwords')
    #nltk.download('punkt')
    stop_words_nltk = nltk.corpus.stopwords.words('english')
    stop_words_nltk = stop_words_nltk + ['','ag_text','cx_text','agtext','cxtext','hi','hello','amazon','please',
                            'itis','thatis','mm','eh','ummm','wo','st','would','name','oh','amazonis',
                            'one','two','five','hermes','emailaddress','chat','nine','orderid','seven',
                            'eight','twenty','six','yet','thereis','ahhh','pounds','dot','url','id','ml','um',
                            'four','bz','thousand','I','mr','•','phonenumber','rd']

    sto_word_list_remove = ['no', 'nor', 'not', 'only','again','against','do', 'should', 'are']

    for word in sto_word_list_remove: 
        stop_words_nltk.remove(word)

    # dropping missing values from text columns alone. 
    df[['title', 'author']] = df[['title', 'author']].fillna(value = 'missing')
    df = df.dropna()
    #df = df[df.wc_news > 20]
    
    print("Shape of df {}".format(df.shape))
    print(sum(df.isnull().sum()))
    print("***** START CLEANING PROCESS PLEASE WAIT THIS TAKES A BIT LONG SORRY !!!...")

    df['clean_title'] = clean_text_dataframe(df, 'title', stop_words_nltk)
    df['clean_text'] = clean_text_dataframe(df, 'text', stop_words_nltk)
    df['clean_author'] = clean_text_dataframe(df, 'author', stop_words_nltk)

    df["wc_heading_clean"] = df["clean_title"].apply(lambda n: len(str(n).split()))
    df["wc_news_clean"] = df["clean_text"].apply(lambda n: len(str(n).split()))

    cols_to_use = ['id','title','author','text','label','clean_title','clean_text','clean_author','wc_heading','wc_news','wc_heading_clean','wc_news_clean']
    df = df[cols_to_use]
    df['ratio_news_clean_orig'] = df.wc_news_clean / df.wc_news
    df['ratio_head_clean_orig'] = df.wc_heading_clean / df.wc_heading
    df = df[df.wc_news_clean > 10]

    return df

####################################################################################
# EMBEDDINGS
####################################################################################

def create_padded_seq (df, type_of_data, cols_to_use_list, output_location, save_label, max_length):
    
    trunc_type = 'pre'
    padding_type = 'pre'
    oov_tok = "<oov>"

    a,t = None, None
    if len(cols_to_use_list) == 1:
        a = np.array(df[cols_to_use_list[0]].astype("str"))
        t = a
        #a = np.concatenate((a1,a2), axis=0)
    else:
        a1 = np.array(df[cols_to_use_list[0]].astype("str"))
        a2 = np.array(df[cols_to_use_list[1]].astype("str"))
        a = np.concatenate((a1,a2), axis=0)
        t = a1+' '+a2
    
    print("***** DONE CONCAT")
    # %%
    
    #if type_of_data == 'TRAIN':
     #   tokenizer = Tokenizer(oov_token = oov_tok)
      #  tokenizer.fit_on_texts(a)
        
       # # save tokenizer pickle
        ##pickle_save_location = ""
        #pickle_save_location = output_location+'/tokenizer_'+save_label+'.pkl'
        ##print("pickle save location: %s" %pickle_save_location)
            
        #print('tokenizer save location: '+str(pickle_save_location))
        #with open(pickle_save_location, 'wb') as handle:
         #   pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # %%
        #word_index = tokenizer.word_index
        #vocab_size = len(word_index) + 1
        #print('******* vocab size is:'+str(vocab_size))
        
        #encoded_docs = tokenizer.texts_to_sequences(t)
        #padded_sequence = pad_sequences(encoded_docs, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        # %%
        ####################################################################################
        # creating matrix for words and encoding
        ####################################################################################
        #embeddings_matrix = np.zeros((vocab_size, embedding_dim))
        #word_not_found = []

        #for word, i in word_index.items():
         #   embedding_vector_learned = emb_learned.get(word)
          #  embedding_vector_glove = emb_glove.get(word)

           # if embedding_vector_learned is not None:
            #    embeddings_matrix[i] = embedding_vector_learned
            #elif embedding_vector_glove is not None:
             #   embeddings_matrix[i] = embedding_vector_glove
            #else:
             #   embeddings_matrix[i] = emb_learned.get('[UNK]')
              #  word_not_found.append(word)

        #print('Shape of Embedding Matrix: '+str(embeddings_matrix.shape))
        #print('Number of words not found in the embeddings: '+str(len(word_not_found)))

        #return padded_sequence, embeddings_matrix, word_not_found, vocab_size
    #else:
    pickle_saved_location = output_location+'/tokenizer_'+save_label+'.pkl'
        
    print('tokenizer saved location for test: '+str(pickle_saved_location))
    
    tokenizer = pickle.load(open(pickle_saved_location,'rb'))
    
    encoded_docs = tokenizer.texts_to_sequences(t)
    padded_sequence = pad_sequences(encoded_docs, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    return padded_sequence


def run_fake_news_code():
    ## -------------------------------------
    ## ----- file paths
    ## -------------------------------------
    current_wd = os.getcwd()
    print("***** CURRENT DIRECTORY: %s" %current_wd)

    file_path_train = current_wd+'/fake_news/train.csv'
    file_path_test = current_wd+'/fake_news/test.csv'
    file_path_labels = current_wd+'/fake_news/labels.csv'

    ## -------------------------------------
    ## ----- file paths END
    ## -------------------------------------

    

    ### ---- READ
    df_train, df_test = create_df(file_path_train, file_path_test, file_path_labels)
    #print("df_train shape {}".format(df_train.shape))
    print("***** df_test shape {}".format(df_test.shape))

    ### ---- CLEAN
    df_test = start_df_cleaning(df_test)
    print("***** DONE WITH CLEANING")

    df_final_test = df_test[df_test.wc_news_clean > 10].copy()
    print("df_test shape {}".format(df_test.shape))

    embedding_dim = 300
    max_length = 900
    save_label = 'lstm_model'
    output_location = current_wd+'/output'

    #### --------
    ## SELF EMBEDDINGS
    #### --------

    #df_vectors = pd.read_csv(current_wd+'/output/vectors_900s_300d.tsv', sep='\t',header=None)
    #df_words = pd.read_csv(current_wd+'/output/metadata_900s_300d.tsv', sep='\t',header=None)

    #word_vect = df_vectors.values
    #word_emb_dict = {}
    #for k,v in enumerate(df_words.values):
     #   word_emb_dict[v[0]] = word_vect[k]

    #### --------
    ## GLOVE EMBEDDINGS
    #### --------
    #embeddings_index = {}
    #glove_embedding_path = current_wd+'/glove.6B.300d.txt'
    #with open(glove_embedding_path) as f:
     #   for line in f:   
      #      values = line.split()
       #     word = values[0]
        #    coefs = np.asarray(values[1:], dtype='float32')
         #   embeddings_index[word] = coefs

    ### ---- PADDING
    df_final_test['combined_clean'] =  df_final_test.clean_author + ' ' + df_final_test.clean_text

    padded_seq = create_padded_seq(df_final_test, 'TEST', 
                                           ['combined_clean'],
                                          output_location, save_label, max_length)

    ### ---- LSTM
    model_lstm = tf.keras.models.load_model(output_location+'/model_lstm_'+save_label+'.h5')
    y_lstm_prob_model = model_lstm.predict(padded_seq)
    df_final_test['lstm_prob'] = y_lstm_prob_model
    print("***** GOT LSTM PROB")

    ### ---- TFIDF
    df_test_tfidf = df_final_test[['combined_clean','wc_heading','wc_news','ratio_news_clean_orig','ratio_head_clean_orig','lstm_prob','label']]
    
    tfidf_vectorizer = pickle.load(open(current_wd+"/output/vectorizer.pickle", "rb"))

    x_test_transform = tfidf_vectorizer.transform(df_test_tfidf.combined_clean)
    x_test = np.concatenate((x_test_transform.todense(), df_test_tfidf[['wc_heading','wc_news','ratio_news_clean_orig','ratio_head_clean_orig','lstm_prob']].values), axis=1)
    y_test = df_test_tfidf['label']

    rf_model = pickle.load(open(current_wd+'/output/random_forest_model.sav', "rb"))
    y_probability = rf_model.predict_proba(x_test)[:,1] ## GET PROBABILITIES

    final_threshold_prob = 0.488
    final_predict = np.where(y_probability>=final_threshold_prob,1,0)
#    fn['predicted'] = fn['predicted'].astype(int)
    cnf_mat = metrics.confusion_matrix(y_test, final_predict)

    print("***** FINAL RESULT")
    print("** ACCURACY")
    print(round(accuracy_score(y_test, final_predict) * 100,3))
    print("** CNF MAT")
    print(cnf_mat)
    

if __name__ == "__main__":
    print("MAIN")
    run_fake_news_code()