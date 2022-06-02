
from modules.packages import *

# importing train data
data=pd.read_csv("data/cleaned.csv",error_bad_lines=False, delimiter='|')
data['content'] = data['content'].astype('U').values
data['type'] = data['type'].astype('U').values


# TF-IDF model 
def TFIDFModels(Model,txt):
    
    x_train, x_test, y_train, y_test = train_test_split(data[['content', 'title']], data['label'], test_size=0.2, random_state=50)
    
    vect = ColumnTransformer([('vectorizer1', TfidfVectorizer(min_df = 5, max_df =0.8, sublinear_tf = True, use_idf = True), 'content'),('vectorizer2', TfidfVectorizer(min_df = 5, max_df =0.8, sublinear_tf = True, use_idf = True), 'title')])

    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    model     = Model
    model.fit(train_vect, y_train)
    predicted = model.predict(test_vect)
    
    accuracy  = model.score(train_vect, y_train)
    predicted = model.predict(test_vect)

    # Save the vectorizer
    vec_file = 'vectorizer/vectorizer_content_title/vectorizer_TFIDF'+ '_' + str(model) + '_' +'_content_title.pickle'
    pickle.dump(vect, open(vec_file, 'wb'))

    # Save the model
    mod_file = 'classification/classification_content_title/classification_TFIDF'+ '_' + str(model) + '_' +'_content_title.model'
    pickle.dump(model, open(mod_file, 'wb'))
    
    report = classification_report(y_test, predicted, output_dict=True)

    with open('output/output_content_title/_TFIDF'+ '_' + str(model) + '_' +'_content_title.txt', "w") as text_file:
        text_file.write(str(report))

    cf_matrix = confusion_matrix(y_test, predicted)

    disp = ConfusionMatrixDisplay(confusion_matrix= cf_matrix) 
    disp.plot()
    
    plt.savefig('figs/fig_content_title/fig_TFIDF'+ '_' + str(model) + '_' +'_content_title.png')
    plt.show(block = False)

    print(txt)
    print('Accuracy score train set :', accuracy)
    print('Accuracy score test set  :', accuracy_score(y_test, predicted),'\n')
    print('Positive: ', report['1'])
    print('Neutral : ', report['0'])
    print(report)
    print('\n -------------------------------------------------------------------------------------- \n')

# TF-IDF model 
def KNN_TFIDF():
    
    x_train, x_test, y_train, y_test = train_test_split(data[['content', 'title']], data['label'], test_size=0.2, random_state=50)
    
    vect = ColumnTransformer([('vectorizer1', TfidfVectorizer(min_df = 5, max_df =0.8, sublinear_tf = True, use_idf = True), 'content'),('vectorizer2', TfidfVectorizer(min_df = 5, max_df =0.8, sublinear_tf = True, use_idf = True), 'title')])

    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    for k in [1,3,5,7,10]:

        model = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
        model.fit(train_vect, y_train)
        predicted = model.predict(test_vect)

        accuracy  = model.score(train_vect, y_train)
        predicted = model.predict(test_vect)

        # Save the vectorizer
        vec_file = 'vectorizer/vectorizer_content_title/vectorizer_KNN_' + str(k) + '_content_title.pickle'
        pickle.dump(vect, open(vec_file, 'wb'))

        # Save the model
        mod_file = 'classification/classification_content_title/classification_KNN_' + str(k) + '_content_title.model'
        pickle.dump(model, open(mod_file, 'wb'))

        report = classification_report(y_test, predicted, output_dict=True)

        with open('output/output_content_title/_KNN_' + str(k) + '_content_title.txt', "w") as text_file:
            text_file.write(str(report))

        cf_matrix = confusion_matrix(y_test, predicted)

        disp = ConfusionMatrixDisplay(confusion_matrix= cf_matrix) 
        disp.plot()
        
        plt.savefig('figs/fig_content_title/fig_KNN_' + str(k) + '_content_title.png')
        plt.show(block = False)
        

        print("Classification Report for k = {} is:\n".format(k))
        print('Accuracy score train set :', accuracy) 
        print('Accuracy score test set  :', accuracy_score(y_test, predicted),'\n')
        print(report)
        print('\n -------------------------------------------------------------------------------------- \n')


# n-gram model 
def NgramModels(Model , txt, n):
    
    x_train, x_test, y_train, y_test = train_test_split(data[['content', 'title']], data['label'], test_size=0.2, random_state=50)
    
    vect = ColumnTransformer([('vectorizer1', CountVectorizer(max_features=1000 , ngram_range=(n,n)), 'content'),('vectorizer2', CountVectorizer(max_features=1000 , ngram_range=(n,n)), 'title')])

    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    model     = Model
    model.fit(train_vect, y_train)
    predicted = model.predict(test_vect)
    
    accuracy  = model.score(train_vect, y_train)
    predicted = model.predict(test_vect)
    
    # Save the vectorizer
    vec_file = 'vectorizer/vectorizer_content_title/vectorizer_ngram_' + str(n) + '_' + str(model) + '_' + '_content_title.pickle'
    pickle.dump(vect, open(vec_file, 'wb'))

    # Save the model
    mod_file = 'classification/classification_content_title/classification_ngram_' + str(n) + '_' + str(model) + '_'+ '_content_title.model'
    pickle.dump(model, open(mod_file, 'wb'))
    

    report = classification_report(y_test, predicted, output_dict=True)

    with open('output/output_content_title/_ngram_' + str(n) + '_' + str(model) + '_'+ '_content_title.txt', "w") as text_file:
            text_file.write(str(report))

    cf_matrix = confusion_matrix(y_test, predicted)

    disp = ConfusionMatrixDisplay(confusion_matrix= cf_matrix) 
    disp.plot()

    plt.savefig('figs/fig_content_title/fig_ngram_' + str(n) + '_' + str(model) + '_'+ '_content_title.png')
    plt.show(block = False)


    print("Models with " , n , "-grams :\n")
    print('********************** \n')
    print(txt)
    print('Accuracy score train set :', accuracy)
    print('Accuracy score test set  :', accuracy_score(y_test, predicted),'\n')
    print(report)
    print('\n --------------------------------------------------------------------------------------------------- \n')

#n-gram KNN
def KNN_Ngram(n):
    
    x_train, x_test, y_train, y_test = train_test_split(data[['content', 'title']], data['label'], test_size=0.2, random_state=50)
    
    vect = ColumnTransformer([('vectorizer1', CountVectorizer(max_features=1000 , ngram_range=(n,n)), 'content'),('vectorizer2', CountVectorizer(max_features=1000 , ngram_range=(n,n)), 'title')])
  
    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    for k in [1,3,5,7,10]:

        model = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
        model.fit(train_vect, y_train)
        predicted = model.predict(test_vect)

        accuracy  = model.score(train_vect, y_train)
        predicted = model.predict(test_vect)

        # Save the vectorizer
        vec_file = 'vectorizer/vectorizer_content_title/vectorizer_ngramKNN_' + str(n) + 'k=' + str(k) + '_content_title.pickle'
        pickle.dump(vect, open(vec_file, 'wb'))

        # Save the model
        mod_file = 'classification/classification_content_title/classification_ngramKNN_' + str(n) + 'k=' + str(k) + '_content_title.model'
        pickle.dump(model, open(mod_file, 'wb'))

        report = classification_report(y_test, predicted, output_dict=True)

        with open('output/output_content_title/_ngramKNN_' + str(n) + 'k=' + str(k) + '_content_title.txt', "w") as text_file:
            text_file.write(str(report))

        cf_matrix = confusion_matrix(y_test, predicted)

        disp = ConfusionMatrixDisplay(confusion_matrix= cf_matrix) 
        disp.plot()

        plt.savefig('figs/fig_content_title/fig_ngramKNN_' + str(n) + 'k=' + str(k) + '_content_title.png')
        plt.show(block = False)


        print("Models with " , n , "-grams :\n")
        print('********************** \n')
        print("Classification Report for k = {} is:\n".format(k))
        print('Accuracy score train set :', accuracy)
        print('Accuracy score test set  :', accuracy_score(y_test, predicted),'\n')
        print(report)
        print('\n -------------------------------------------------------------------------------------- \n')



# n-gram models 
NgramModels(Model=LogisticRegression(solver = 'sag'),txt='Logistic Regression Model : \n ', n=2)

NgramModels(Model=LogisticRegression(solver="saga", n_jobs=-1, max_iter=10000),txt='Logistic Regression Model : \n ', n=3)

NgramModels(Model=svm.SVC(kernel='linear') ,txt='Support Vectoer Classifier Model : \n ', n=2)

NgramModels(Model=svm.SVC(kernel='linear') ,txt='Support Vectoer Classifier Model : \n ', n=3)

NgramModels(Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ', n=2)

NgramModels(Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ', n=3)

KNN_Ngram(2)

KNN_Ngram(3)

# TF-IDF models 
TFIDFModels(Model=LogisticRegression(solver='sag'),txt='Logistic Regression Model : \n ')

TFIDFModels(Model=svm.SVC(kernel='linear'),txt='Support Vector Classifier Model : \n ')

TFIDFModels(Model=tree.DecisionTreeClassifier(),txt='Decision Tree Classifier Model : \n ')

KNN_TFIDF()
