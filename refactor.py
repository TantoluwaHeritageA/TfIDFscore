import nltk
import numpy as np
import pandas as pd
import math
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')
# removing stopwords
all_stopwords = stopwords.words('english')

search_results = {
    "doc1": ["python" , "vs" , "r" , "for" , "data" , "science" , "project"] ,
    "doc2": ["python" , "vs" , "r" , "know" , "the" , "difference"] ,
    "doc3": ["which" , "is" , "better" , "for" , "data" , "analysis" , "r" , "or" , "python"]
}


def remove_stopwords():
    result_dict = {}
    for key in search_results.keys():
        if key not in result_dict:
            result_dict[key] = []
        for item in search_results[key]:
            if not item in all_stopwords:
                result_dict[key].append(item)
    #  returns terms with stopwords removed
    doc1 = result_dict["doc1"]
    doc2 = result_dict["doc2"]
    doc3 = result_dict["doc3"]
    return doc1 , doc2 , doc3


# returns count of each term

def word_counter():
    d1 , d2 , d3 = remove_stopwords()
    word_count1 = dict(Counter(d1))
    word_count2 = dict(Counter(d2))
    word_count3 = dict(Counter(d3))
    return word_count1 , word_count2 , word_count3


# returns total count of terms in each
def count_total_terms():
    count_document = {}
    for key in search_results.keys():
        count_document[key] = len(search_results[key])
    return count_document


# calculate term frequency
def calculate_termfreq():
    wc1 , wc2 , wc3 = word_counter()
    total_count_doc = count_total_terms()
    tf_1 = {key: round((wc1[key] / total_count_doc['doc1']) , 3) for key in wc1}
    tf_2 = {key: round((wc2[key] / total_count_doc['doc2']) , 3) for key in wc2}
    tf_3 = {key: round((wc3[key] / total_count_doc['doc3']) , 3) for key in wc3}
    tf_table1 = pd.DataFrame.from_dict(tf_1,orient='index', columns=["weight of term in doc"] )
    print("__________________________________________")
    print("Doc 1")
    print(tf_table1)
    tf_table2 = pd.DataFrame.from_dict(tf_2,orient='index', columns=["weight of term in doc"] )
    print("__________________________________________")
    print("Doc 2")
    print(tf_table2)
    tf_table3 = pd.DataFrame.from_dict(tf_3,orient='index', columns=["weight of term in doc"] )
    print("__________________________________________")
    print("Doc 3")
    print(tf_table3)
    return tf_1 , tf_2 , tf_3


# calculate inverse term frequency
def calculate_idf():
    wc1 , wc2 , wc3 = word_counter()
    total_count_doc = count_total_terms()
    idf_1_sol = {key: math.log(len(total_count_doc) / wc1[key]) for key in wc1}
    idf_2_sol = {key: math.log(len(total_count_doc) / wc2[key]) for key in wc2}
    idf_3_sol = {key: math.log(len(total_count_doc) / wc3[key]) for key in wc3}
    idf1 = pd.DataFrame.from_dict(idf_1_sol,orient='index', columns=["idf"] )
    # print("__________________________________________")
    # print("Doc 1")
    # print(idf1)
    # idf2 = pd.DataFrame.from_dict(idf_2_sol,orient='index', columns=["idf"] )
    # print("__________________________________________")
    # print("Doc 2")
    # print(idf2)
    # idf3 = pd.DataFrame.from_dict(idf_3_sol,orient='index', columns=["idf"] )
    # print("__________________________________________")
    # print("Doc 3")
    # print(idf3)
    return idf_1_sol , idf_2_sol , idf_3_sol


# calculate term frequency - inverse document frequency score
def tf_idf_scoring():
    idf1_val , idf2_val , idf3_val = calculate_idf()
    tf1_val , tf2_val , tf3_val = calculate_termfreq()
    tf_idf1 = {key: round(tf1_val[key] * idf1_val[key_1] , 3) for key in tf1_val for key_1 in idf1_val}
    tf_idf_2 = {key: round(tf2_val[key] * idf2_val[key_1] , 3) for key in tf2_val for key_1 in idf2_val}
    tf_idf_3 = {key: round(tf3_val[key] * idf3_val[key_1] , 3) for key in tf3_val for key_1 in idf3_val}
    # tfidf1 = pd.DataFrame.from_dict(tf_idf1,orient='index', columns=["tf-idf score"] )
    # print("__________________________________________")
    # print("Doc 1")
    # print(tfidf1)
    # tfidf2 = pd.DataFrame.from_dict(tf_idf_2,orient='index', columns=["tf-idf score"] )
    # print("__________________________________________")
    # print("Doc 2")
    # print(tfidf2)
    # tfidf3 = pd.DataFrame.from_dict(tf_idf_3,orient='index', columns=["tf-idf score"] )
    # print("__________________________________________")
    # print("Doc 3")
    # print(tfidf3)
    return tf_idf1 , tf_idf_2 , tf_idf_3


# document with tf-idf scores
def tf_idf_document():
    tf_idf1 , tf_idf_2 , tf_idf_3 = tf_idf_scoring()
    search_results.update({"doc1": tf_idf1 , "doc2": tf_idf_2 , "doc3": tf_idf_3})
    # converted to dataframes
    doc_with_tf_idf = pd.DataFrame(search_results)
    doc_with_tf_idf.replace(np.nan , 0.00 , inplace=True)
    print(doc_with_tf_idf)
    return doc_with_tf_idf


# searching keywords
def search_keywords():
    doc_with_tf_idf = tf_idf_document()
    query = str(input("enter a query to search: ")).lower()
    query1 = str(input("enter a query to search: ")).lower()
    # query2 = str(input("enter a query to search: ")).lower()
    # search one keyword
    # for k , v in doc_with_tf_idf.loc[query].iteritems():
    #     print(f"{k}     :       {v} ")
    # search for two, three keywords using "or" operator
    # print(f"To be searched: {query} or {query1}")
    # # print(f"To be searched: {query} or {query1} or {query2}")
    # result_with_or = doc_with_tf_idf.loc[doc_with_tf_idf.index.str.startswith((query,query1))]
    # # result_with_or = doc_with_tf_idf.loc[doc_with_tf_idf.index.str.startswith((query,query1, query2))]
    # print(result_with_or)
    # # search for two or three keywords using "and" operator

    print(f"To be searched: {query} and {query1}")
    result_with_and = doc_with_tf_idf.loc[doc_with_tf_idf.index.str.startswith(query) & doc_with_tf_idf.index.str.startswith(query1, na=True)]
    print(result_with_and)


search_keywords()


