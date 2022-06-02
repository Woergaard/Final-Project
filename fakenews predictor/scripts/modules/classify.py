from modules.packages import *

# Load the classification model from disk and use for predictions
def classify(utt, modeldata,model ):
    # load the vectorizer
    loaded_vectorizer = pickle.load(open("vectorizer/vectorizer_" + modeldata + "/vectorizer" + model + ".pickle", 'rb'))

    # load the model
    loaded_model = pickle.load(open("classification/classification_" + modeldata + "/classification" + model + ".model", 'rb'))

    # make a prediction
    result = loaded_model.predict(loaded_vectorizer.transform(utt))
    print(result)

    result=pd.DataFrame(result)

    return result
    