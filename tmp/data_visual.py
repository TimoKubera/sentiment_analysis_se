import pandas as pd
import numpy
from matplotlib import pyplot as plt
import random

data = pd.read_csv('./data/github_gold.csv', delimiter=';')
data.drop(columns='ID', inplace=True)

neutral, negative, positive = data['Polarity'].value_counts()

plt.style.use('seaborn-paper')


fig, ax2 = plt.subplots(nrows=1, ncols=1)

"""
ax1.bar(x='Negative', height = negative, color='#F96167', edgecolor='#201E20')
ax1.text(x ='Negative' , y=1500, s='{perc:.1f} %'.format(perc= negative / (negative + neutral + positive)*100), horizontalalignment='center')

ax1.bar(x='Neutral', height = neutral, color='#E7E8D1', edgecolor='#201E20')
ax1.text(x = 'Neutral', y=2500, s='{perc:.1f} %'.format(perc= neutral / (negative + neutral + positive)*100), horizontalalignment='center')

ax1.bar(x='Positive', height = positive, color='#317773', edgecolor='#201E20')
ax1.text(x = 'Positive', y=1500, s='{perc:.1f} %'.format(perc= positive / (negative + neutral + positive)*100), horizontalalignment='center')

ax1.set(xlabel='Polarity', 
       ylabel='Quantity',
       title='Distribution of the polarity classes')
"""

x = [random.randint(0, 100) for i in range(100)]
x += [random.randint(300, 1000) for i in range(100)]

y = [random.randint(0, 100) for i in range(100)]
y += [random.randint(-10, 20) for i in range(100)]

print(x)
print(y)

ax2.scatter(x=x[:100], y=y[:100], marker='x', color='orange', alpha=0.55)
ax2.scatter(x=x[100:], y=y[100:], marker='x', color='blue', alpha=0.55)

plt.show()