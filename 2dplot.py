# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:52:18 2021

@author: asrinutku
"""
import matplotlib.pyplot as plt

y = [1,2,3,4,5,6,7]

x = [0.7871649195766028, 0.8076100371045793, 0.8179527183248526, 0.8247598698250219, 0.8317965185470579, 0.8335784925618683, 0.8348753657738703]

x1 = [0.7752967583763222, 0.7832302767417892, 0.8133779024587052, 0.8374151926054145, 0.8384975427612911, 0.8426161448223314, 0.8454119128158647]


a = plt.scatter(y, x,color = "red")
b = plt.scatter(y, x1,color = "blue")

plt.legend((a,b),
           ('United States', 'Canada'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)


plt.xlabel("K SAYISI")
plt.ylabel("TERS ORANTILI BENZERLİK")
plt.title("K SAYISI VE BENZERLİK ORANI GRAFİĞİ")
plt.show()
print("plot edildi")     