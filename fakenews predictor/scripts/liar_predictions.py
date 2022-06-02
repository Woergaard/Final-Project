from numpy import block
from modules.packages import *
from modules.clean_text import *
from modules.classify import *
from modules.confusion_matrix_liar import *

#############################################################################################
#############################################################################################
########################################### Data ############################################
#############################################################################################
#############################################################################################


# Importing LIAR dataset 
LIAR=pd.read_csv("data/liar.csv",error_bad_lines=False, delimiter = "|")
LIAR.columns = ['id', 'type', 'content', 'title', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10', 'id11']

# Cleaning content column
LIAR.content=[Clean(x) for x in LIAR.content]

# Reducing to essential columns 
LIAR = LIAR[['content', 'type', 'title']]
LIAR['content'] = LIAR['content'].astype('U').values
LIAR['title'] = LIAR['title'].astype('U').values

# Creating label for FAKE/REAL
LIAR['label'] = [1 if LIAR['type'].iloc[x] in ['false','pants-fire', 'fake'] else 0 for x in range(len(LIAR['type']))]#LIAR['type'].map(lambda x: 1 if ['false','pants-fire'] in x else 0)


# Model list 
liste = ["_TFIDF_LogisticRegression(solver='sag')_", 
        "_ngram_2_LogisticRegression(solver='sag')_",
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

#############################################################################################
#############################################################################################
####################################### With content ########################################
#############################################################################################
#############################################################################################

for i in liste: 

    LIARpredict = classify(LIAR['content'], "content", i + "_content")
    

    LIARpredict.columns = ['predicted']
    df = pd.concat([LIAR, LIARpredict], axis=1)
    
    cm = np.array(confusion_matrix_liar(df, 'label', 'predicted'))
    report = classification_report(df['label'], df['predicted'], output_dict=True, zero_division = 0)

    with open("output/output_liar_content/"+ i + "_content.txt", "w") as text_file:
        text_file.write(str(report))

    plt.figure()
    plot_confusion_matrix(cm, classes = [0,1])

    plt.savefig('figs/fig_liar_content/' + i + '.png')
    plt.show(block = False)



#############################################################################################
#############################################################################################
############################### With content and metadata ###################################
#############################################################################################
#############################################################################################

for i in liste: 
   
    LIARpredict = classify(LIAR[['content', 'title']], "content_title", i + "_content_title")

    LIARpredict.columns = ['predicted']
    df = pd.concat([LIAR, LIARpredict], axis=1)

    cm = np.array(confusion_matrix_liar(df, 'label', 'predicted'))
    report = classification_report(df['label'], df['predicted'], output_dict=True, zero_division = 0)

    with open("output/output_liar_content_title/"+ i + "_content_title.txt", "w") as text_file:
        text_file.write(str(report))

    plt.figure()
    plot_confusion_matrix(cm, classes = [0,1])

    plt.savefig('figs/fig_liar_content_title/' + i + '.png')
    plt.show(block = False)



#############################################################################################
#############################################################################################
#################################### Without content ########################################
#############################################################################################
#############################################################################################

for i in liste: 
   
    LIARpredict = classify(LIAR['content'], "title", i +  "_title")

    LIARpredict.columns = ['predicted']
    df = pd.concat([LIAR, LIARpredict], axis=1)

    cm = np.array(confusion_matrix_liar(df, 'label', 'predicted'))
    report = classification_report(df['label'], df['predicted'], output_dict=True, zero_division = 0)

    with open("output/output_liar_title/"+ i + "_title.txt", "w") as text_file:
        text_file.write(str(report))

    plt.figure()
    plot_confusion_matrix(cm, classes = [0,1])

    plt.savefig('figs/fig_liar_title/' + i + '.png')
    plt.show(block = False)
