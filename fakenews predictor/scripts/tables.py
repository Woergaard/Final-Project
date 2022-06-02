from modules.packages import *

# Model list 
liste = ["_ngram_2_LogisticRegression(solver='sag')_",
        "_ngram_3_LogisticRegression(max_iter=10000, n_jobs=-1, solver='saga')_", 
        "_ngram_2_SVC(kernel='linear')_", 
        "_ngram_3_SVC(kernel='linear')_", 
        "_ngram_2_DecisionTreeClassifier()_", 
        "_ngram_3_DecisionTreeClassifier()_", 
        "_ngramKNN_2k=1", 
        "_ngramKNN_2k=3", 
        "_ngramKNN_2k=5", 
        "_ngramKNN_2k=7", 
        "_ngramKNN_2k=10", 
        "_ngramKNN_3k=1", 
        "_ngramKNN_3k=3", 
        "_ngramKNN_3k=5", 
        "_ngramKNN_3k=7", 
        "_ngramKNN_3k=10", 
        "_TFIDF_LogisticRegression(solver='sag')_", 
        "_TFIDF_SVC(kernel='linear')_", 
        "_TFIDF_DecisionTreeClassifier()_", 
        "_KNN_1", 
        "_KNN_3", 
        "_KNN_5", 
        "_KNN_7", 
        "_KNN_10"
        ]

# FAKENEWS_CONTENT

new = pd.DataFrame()

for i in liste: 
    output = pd.DataFrame(pd.read_csv("output/output_content/"+ i + "_content" + ".txt",sep='|',header=None).iloc[:,0].apply(ast.literal_eval).tolist())
    output.index = [i]
    new = new.append(output)       
    

new = pd.concat([new, new["macro avg"].apply(pd.Series)], axis=1)
new = pd.concat([new, new["weighted avg"].apply(pd.Series)], axis=1)

new = new.drop(columns=["0", "1", "macro avg", "weighted avg"])
new.columns = ['accuracy' , 'precision macro','recall macro', 'f1-score macro', 'support macro','precision weighted', 'recall weighted', 'f1-score weighted', 'support weighted']

#display(new)

with open('tables/table_content.tex', 'w') as tf:
     tf.write(new.to_latex())

# FAKENEWS_CONTENT_TITLE

new = pd.DataFrame()

for i in liste: 
    output = pd.DataFrame(pd.read_csv("output/output_content_title/"+ i + "_content_title" + ".txt",sep='|',header=None).iloc[:,0].apply(ast.literal_eval).tolist())
    output.index = [i]
    new = new.append(output)       
    

new = pd.concat([new, new["macro avg"].apply(pd.Series)], axis=1)
new = pd.concat([new, new["weighted avg"].apply(pd.Series)], axis=1)

new = new.drop(columns=["0", "1", "macro avg", "weighted avg"])
new.columns = ['accuracy' , 'precision macro','recall macro', 'f1-score macro', 'support macro','precision weighted', 'recall weighted', 'f1-score weighted', 'support weighted']

#display(new)

with open('tables/table_content_title.tex', 'w') as tf:
     tf.write(new.to_latex())

# FAKENEWS_TITLE 

new = pd.DataFrame()

for i in liste: 
    output = pd.DataFrame(pd.read_csv("output/output_title/"+ i + "_title" + ".txt",sep='|',header=None).iloc[:,0].apply(ast.literal_eval).tolist())
    output.index = [i]
    new = new.append(output)       
    

new = pd.concat([new, new["macro avg"].apply(pd.Series)], axis=1)
new = pd.concat([new, new["weighted avg"].apply(pd.Series)], axis=1)

new = new.drop(columns=["0", "1", "macro avg", "weighted avg"])
new.columns = ['accuracy' , 'precision macro','recall macro', 'f1-score macro', 'support macro','precision weighted', 'recall weighted', 'f1-score weighted', 'support weighted']

#display(new)

with open('tables/table_title.tex', 'w') as tf:
     tf.write(new.to_latex())

# LIAR_CONTENT

new = pd.DataFrame()

for i in liste: 
    output = pd.DataFrame(pd.read_csv("output/output_liar_content/"+ i + "_content" + ".txt",sep='|',header=None).iloc[:,0].apply(ast.literal_eval).tolist())
    output.index = [i]
    new = new.append(output)       
    

new = pd.concat([new, new["macro avg"].apply(pd.Series)], axis=1)
new = pd.concat([new, new["weighted avg"].apply(pd.Series)], axis=1)

new = new.drop(columns=["0", "1", "macro avg", "weighted avg"])
new.columns = ['accuracy' , 'precision macro','recall macro', 'f1-score macro', 'support macro','precision weighted', 'recall weighted', 'f1-score weighted', 'support weighted']

#display(new)

with open('tables/table_liar_content.tex', 'w') as tf:
     tf.write(new.to_latex())


# LIAR_CONTENT_TITLE 

new = pd.DataFrame()

for i in liste: 
    output = pd.DataFrame(pd.read_csv("output/output_liar_content_title/"+ i + "_content_title" +  ".txt",sep='|',header=None).iloc[:,0].apply(ast.literal_eval).tolist())
    output.index = [i]
    new = new.append(output)       
    

new = pd.concat([new, new["macro avg"].apply(pd.Series)], axis=1)
new = pd.concat([new, new["weighted avg"].apply(pd.Series)], axis=1)

new = new.drop(columns=["0", "1", "macro avg", "weighted avg"])
new.columns = ['accuracy' , 'precision macro','recall macro', 'f1-score macro', 'support macro','precision weighted', 'recall weighted', 'f1-score weighted', 'support weighted']

#display(new)

with open('tables/table_liar_content_title.tex', 'w') as tf:
     tf.write(new.to_latex())


# LIAR_TITLE 

new = pd.DataFrame()

for i in liste: 
    output = pd.DataFrame(pd.read_csv("output/output_liar_title/"+ i + "_title" + ".txt",sep='|',header=None).iloc[:,0].apply(ast.literal_eval).tolist())
    output.index = [i]
    new = new.append(output)       
    

new = pd.concat([new, new["macro avg"].apply(pd.Series)], axis=1)
new = pd.concat([new, new["weighted avg"].apply(pd.Series)], axis=1)

new = new.drop(columns=["0", "1", "macro avg", "weighted avg"])
new.columns = ['accuracy' , 'precision macro','recall macro', 'f1-score macro', 'support macro','precision weighted', 'recall weighted', 'f1-score weighted', 'support weighted']

#display(new)

with open('tables/table_liar_title.tex', 'w') as tf:
     tf.write(new.to_latex())




