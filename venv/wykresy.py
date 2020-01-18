from matplotlib import pyplot as plt
import numpy as np
import csv
from collections import Counter
import pandas as pd

def rysuj():
    data = pd.read_csv('data.csv')
    ids = data['Responder_id']
    lang_resposnes = data['LanguagesWorkedWith']

    language_counter = Counter()

    for lang in lang_resposnes:
        language_counter.update(lang.split(';'))

    languages, popularity = map(list, zip(*language_counter.most_common(15)))

    plt.barh(languages, popularity)
    plt.title('Languages Popularity')
    plt.ylabel('Languages')
    plt.xlabel('Number of people who use')

    return plt

# to samo co wyżej w jednej linijce
# languages = []
# popularity = []
#
# for item in language_counter.most_common(15):
#     languages.append(item[0])
#     popularity.append(item[1])


# ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
#
# x_indexes = np.arange(len(ages_x))
# width = 0.25
#
# dev_y = [38496, 42000, 46752, 49320, 53200,
#          56000, 62316, 64928, 67317, 68748, 73752]
#
# py_dev_y = [45372, 48876, 53850, 57287, 63016,
#             65998, 70003, 70000, 71496, 75370, 83640]
#
# js_dev_y = [37810, 43515, 46823, 49293, 53437,
#             56373, 62375, 66674, 68745, 68746, 74583]
#
# plt.bar(x_indexes - width, dev_y, width = 0.25, label='All Devs')
# plt.bar(x_indexes, py_dev_y, width = 0.25, label='Python')
# plt.bar(x_indexes + width, js_dev_y, width = 0.25, label='JS')
#
# plt.xticks(ticks=x_indexes, labels=ages_x)
#
# plt.xlabel("Ages")
# plt.ylabel("Median Salary")
# plt.title("Median Salary by Age")
#
# plt.style.use('seaborn-deep')
# plt.legend()
#
# plt.grid()
# plt.show()