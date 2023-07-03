from bs4 import BeautifulSoup
from pynliner import Pynliner

f = open("types-of-sequential-data.html", "r+")

# Reading the file
index = f.read()
  
# Creating a BeautifulSoup object and specifying the parser
S = BeautifulSoup(index, 'lxml')


formatted = Pynliner().from_string(str(S)).run()
print(formatted)

f.write(formatted)
f.close()