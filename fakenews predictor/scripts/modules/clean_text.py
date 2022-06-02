from modules.packages import *

# Define Clean Function to fix text
def Clean(text):

  # Frist converting all letters to lower case
  text= text.lower()
  
  # removing unwanted digits ,special chracters from the text
  text= ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", text).split()) #tags
  text= ' '.join(re.sub("^@?(\w){1,15}$", " ", text).split())
    
  text= ' '.join(re.sub("(\w+:\/\/\S+)", " ", text).split())   #Links
  text= ' '.join(re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"," ", text).split()) 
  text= ' '.join(re.sub(r'http\S+', '',text).split())
  
  
  text= ' '.join(re.sub(r'www\S+', '',text).split())
  text= ' '.join(re.sub("\s+", " ",text).split()) #Extrem white Space
  text= ' '.join(re.sub("[^-9A-Za-z ]", "" ,text).split()) #digits 
  text= ' '.join(re.sub('-', ' ', text).split()) 
  text= ' '.join(re.sub('_', ' ', text).split()) #underscore 

  # removing stopwards and numbers from STRING library
  table= str.maketrans('', '', string.punctuation+string.digits)
  text = text.translate(table)
  
  # Split Sentence as tokens words 
  tokens = word_tokenize(text)
  
  # converting words to their root forms by STEMMING THE WORDS 
  stemmed = [porter.stem(word) for word in tokens] # Covert words to their rootbut not actual
  
  # Delete each stop words from English stop words
  words = [w for w in stemmed if not w in n_words] #n_words contains English stop words

  text  = ' '.join(words)
    
  return text