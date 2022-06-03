from modules.packages import *
from modules.clean_text import *
from modules.classify import *
from modules.confusion_matrix_liar import *


# importing train data
data=pd.read_csv("data/1mio-raw.csv",error_bad_lines=False, nrows=1000000)
data = data.drop_duplicates(subset = ['content'])

# dropping the id, title and author column
data=data[['type', 'content', 'title']]

#checking the null in data
data.isnull().sum() 

# dropping Na values
data = data.dropna()
data = data.reset_index(drop = True)

data = data.drop_duplicates(subset = ['content'])
data = data.reset_index(drop = True)

data = data.sample(75000, random_state=44)
data = data.reset_index(drop = True)

data['label'] = [1 if data['type'].iloc[x] in ['fake','conspiracy','unreliable','junksci', 'rumor'] else 0 for x in range(len(data['type']))]

# apply Clean Funsction to our Text
data.content=[Clean(x) for x in data.content]
data.title=[Clean(x) for x in data.title]

data.to_csv("data/cleaned.csv", sep ='|')
