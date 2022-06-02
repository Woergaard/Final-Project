import os

# Run models
#os.system("python3 scripts/data_cleaning.py")
#os.system("python3 scripts/model_content.py")
#os.system("python3 scripts/model_content_title.py")
#os.system("python3 scripts/model_title.py")

# Run model predictions on LIAR dataset 
#os.system("python3 scripts/liar_predictions.py") 

# Create tables of data
os.system("python3 scripts/tables.py") 

# Create .tex file with tables and figures 
os.chdir('tex')
os.system("pdflatex main.tex") 


