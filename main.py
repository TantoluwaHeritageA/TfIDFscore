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

document = pd.DataFrame(dict([(key , pd.Series(value)) for key , value in search_results.items()]))
document.replace(np.nan , 0 , inplace=True)
# print(document)

# removing stop words from search result
result_dict = {}
for key in search_results.keys():
    if key not in result_dict:
        result_dict[key] = []
    for item in search_results[key]:
        if not item in all_stopwords:
            result_dict[key].append(item)

doc1 = result_dict["doc1"]
doc2 = result_dict["doc2"]
doc3 = result_dict["doc3"]
# search_results.update({"doc1": doc1 , "doc2": doc2 , "doc3": doc3})
# # print(search_results)
# # converted to dataframes
# doc_with_sw = pd.DataFrame(dict([(key , pd.Series(value)) for key , value in search_results.items()]))
# doc_with_sw.replace(np.nan , 0.00 , inplace=True)
# print(doc_with_sw)

# counting word occurrence in each document
word_count1 = dict(Counter(doc1))
word_count2 = dict(Counter(doc2))
word_count3 = dict(Counter(doc3))
print("DOC 1 with term frequency")
doc1_table = pd.DataFrame.from_dict(word_count1, orient='index', columns=['word_count'])
print(doc1_table)
print("________________________________________")
print("DOC 2 with term frequency")
doc2_table = pd.DataFrame.from_dict(word_count2, orient='index', columns=['word_count'])
print(doc2_table)
print("________________________________________")
print("DOC 3 with term frquency")
doc3_table = pd.DataFrame.from_dict(word_count3, orient='index', columns=['word_count'])
print(doc3_table)
print("________________________________________")

# counting total words in each document
count_document = {}
for key in search_results.keys():
    count_document[key] = len(search_results[key])
print(count_document)
table = pd.DataFrame.from_dict(count_document,orient='index', columns=["Total num of terms in doc"] )
print(table)

# finding term frequency by dividing occurrence of word by the total document
print(word_count1)
# for key in word_count1:
#     new_tf = round((word_count1[key] / count_document['doc1']), 3)
#     print(key,new_tf)

tf_1 = {key:round((word_count1[key] / count_document['doc1']), 3) for key in word_count1}
tf_table1 = pd.DataFrame.from_dict(tf_1,orient='index', columns=["weight of term in doc"] )
print("__________________________________________")
print("Doc 1")
print(tf_table1)

tf_2 = {key:round((word_count2[key] / count_document['doc2']), 3) for key in word_count2}
tf_table2 = pd.DataFrame.from_dict(tf_2,orient='index', columns=["weight of term in doc"] )
print("__________________________________________")
print("Doc 2")
print(tf_table2)
tf_3 = {key:round((word_count3[key] / count_document['doc3']), 3) for key in word_count3}
tf_table3 = pd.DataFrame.from_dict(tf_3,orient='index', columns=["weight of term in doc"] )
print("__________________________________________")
print("Doc 3")
print(tf_table3)

# finding inverse document frequency
# with formula
print(len(count_document))
idf_1_sol = {key:math.log( len(count_document)/word_count1[key] ) for key in word_count1 }
idf1 = pd.DataFrame.from_dict(idf_1_sol,orient='index', columns=["idf"] )
print("__________________________________________")
print("Doc 1")
print(idf1)

idf_2_sol = {key:math.log( len(count_document)/word_count2[key] ) for key in word_count2 }
idf2 = pd.DataFrame.from_dict(idf_2_sol,orient='index', columns=["idf"] )
print("__________________________________________")
print("Doc 2")
print(idf2)

idf_3_sol = {key:math.log( len(count_document)/word_count3[key] ) for key in word_count3 }
idf3 = pd.DataFrame.from_dict(idf_3_sol,orient='index', columns=["idf"] )
print("__________________________________________")
print("Doc 3")
print(idf3)
print("__________________________________________")

# print(tf_1)
# print(idf_1_sol)
# for key in tf_1:
#     # print(tf_1[key])
#     for key1 in idf_1_sol:
#         # print(idf_1_sol[key1])
#
#         print(key, round((tf_1[key] * idf_1_sol[key1]), 2))
# tf-idf scoring, tf * idf

tf_idf1 = {key : round(tf_1[key] * idf_1_sol[key_1], 3 )for key in tf_1 for key_1 in idf_1_sol}
print(tf_idf1)
print("________________________________________")
tf_idf_2 = {key : round(tf_2[key] * idf_2_sol[key_1], 3 )for key in tf_2 for key_1 in idf_2_sol}
print(tf_idf_2)
print("________________________________________")
tf_idf_3 = {key : round(tf_3[key] * idf_3_sol[key_1], 3 )for key in tf_3 for key_1 in idf_3_sol}
print(tf_idf_3)
print("________________________________________")

# updated the main dictionary with all the tf-idf scores
search_results.update({"doc1":tf_idf1, "doc2": tf_idf_2, "doc3": tf_idf_3} )
print(search_results)
# converted to dataframes
doc_with_tf_idf = pd.DataFrame(search_results)
doc_with_tf_idf.replace(np.nan , 0.00 , inplace=True)
print(doc_with_tf_idf)
# searching queries
query = str(input("enter a query to search: ")).lower()
query1 = str(input("enter a query to search: ")).lower()
query2 = str(input("enter a query to search: ")).lower()
# search one keyword
# for k, v in doc_with_tf_idf.loc[query].iteritems():
#     print(f"{k}     :       {v} ")

# search for two, three keywords using "or" operator
print(f"To be searched: {query} or {query1}")
# print(f"To be searched: {query} or {query1} or {query2}")
result_with_or = doc_with_tf_idf.loc[doc_with_tf_idf.index.str.startswith((query,query1))]
# result_with_or = doc_with_tf_idf.loc[doc_with_tf_idf.index.str.startswith((query,query1, query2))]
print(result_with_or)
# search for two or three keywords using "and" operator

print(f"To be searched: {query} and {query1}")
result_with_and = doc_with_tf_idf.loc[doc_with_tf_idf.index.str.startswith(query) & doc_with_tf_idf.index.str.startswith(query1, na=True)]
print(result_with_and)











# term1_list = []
# term2_list = []
# term3_list = []
#
# for key in data.keys():
#     if key == "term1":
#         for item in data[key]:
#             if not item in all_stopwords:
#                 term1_list.append(item)
#     elif key == "term2":
#         for item in data[key]:
#             if not item in all_stopwords:
#                 term2_list.append(item)
#     else:
#         for item in data[key]:
#             if not item in all_stopwords:
#                 term3_list.append(item)
#
# print(term1_list)
# print(term2_list)
# print(term3_list)


# for key in data:
#     print(data[key])
#     for item in data[key]:
#         if not item in all_stopwords:
#             print(item.split())

# values = data[key]
# for item in values:
#     if not item in all_stopwords:
#         print(item)

#
# for items in search_results.values():
#     for val in items:
#         if not val in all_stopwords:
#             print(val)

