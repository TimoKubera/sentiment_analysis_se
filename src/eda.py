import os
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.abspath(os.path.join(__file__, "../.."))
data = pd.read_csv(os.path.join(path, "data/github_gold.csv"), delimiter=";")

print("Preview: ")
print(data.head())
print("\n")

print("Stats: ")
print(data.Polarity.value_counts())

# Balkendiagramm
ax = plt.axes()
ax.hist(data.Polarity, bins=25)

ax.set(xlabel='Petal Length (cm)', 
       ylabel='Frequency',
       title='Distribution of Petal Lengths')