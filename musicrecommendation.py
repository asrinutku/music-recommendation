# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:08:04 2020

@author: asrinutku
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



#%% display işlemi
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#%%
def tekraronleme(df,columns):
    
    if not df[df.duplicated(columns)].empty:
        initial_rows = df.shape[0]

        print ('Orjinal df boyutları {0}'.format(df.shape))
        df = df.drop_duplicates(columns)
        current_rows = df.shape[0]
        print ('Yeni df boyutları {0}'.format(df.shape))
        print ('{0} satır silindi '.format(initial_rows - current_rows))
    else : 
        print("DF ' de tekrar eden veri bulunmadı . ")
        
    
        
#%%
def plotting(x,y):
    
    # k kaç adet şarkı oldugunu gosteriyor
    # x benzerlik oranlari
    
    
    plt.scatter(y, x,zorder=1)
    plt.xlabel("K SAYISI")
    plt.ylabel("TERS ORANTILI BENZERLİK")
    plt.title("K SAYISI VE BENZERLİK ORANI GRAFİĞİ")
    plt.show()
    print("plot edildi")     
  
    
    
#%%

def sarki_sorgula(aranan_sarki, matrix_binary, knn_model_binary, k):
    
    
    aranan_index = 0
    eslesensarkilar = []
    
    for i in matrix_binary.index:
        oran = fuzz.ratio(i.lower(), aranan_sarki.lower())
        if oran >= 40:
            index = matrix_binary.index.tolist().index(i)
            eslesensarkilar.append((i, oran, index))
        

    aranan_index = max(eslesensarkilar, key = lambda x: x[1])[2] 
    
    if (aranan_index== 0):
        print("Aradıgınız sarkı bulunamadı")
        return 0
    
    distances, indices = knn_model_binary.kneighbors(matrix_binary.iloc[aranan_index, :].values.reshape(1, -1), n_neighbors = k + 1)
    benzerlikoranlari = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print("Aradıgınız sarkı ",(matrix_binary.index[aranan_index])," icin oneriler:\n")
        else:
            print(i,":",matrix_binary.index[indices.flatten()[i]], "--", distances.flatten()[i], "benzerlik oranı ile")
        
            benzerlikoranlari.append(distances.flatten()[i])
        
    

    print("BENZERLİK:",benzerlikoranlari)
    print("K:",k)
    y = list(range(k))
    print("Y:",y)
    
    plotting(benzerlikoranlari,y)
    
      
    return 0

# %%

kullanici_sarki_info = pd.read_table('usersha1-artmbid-artname-plays.tsv',
                          header = None, nrows = 2e7,
                          names = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols = ['users', 'artist-name', 'plays'])
kullanici_info = pd.read_table('usersha1-profile.tsv',
                          header = None,
                          names = ['users', 'gender', 'age', 'country', 'signup'],
                          usecols = ['users', 'country'])


# %%
print("temizlikten once kullanici_sarki_info tablomuzdaki satır sayisi  : "  , kullanici_sarki_info.shape[0])

if kullanici_sarki_info['artist-name'].isnull().sum() > 0:
    kullanici_sarki_info = kullanici_sarki_info.dropna(axis = 0, subset = ['artist-name'])
    
print("temizlikten sonra kullanici_sarki_info tablomuzdaki satır sayisi : " , kullanici_sarki_info.shape[0] ,
      "\nEger arada bir fark yoksa eksik datamız yok demektir ")
   
#%%

# gruplama islemi

sarki_calinmasayisi = (kullanici_sarki_info.
     groupby(by = ['artist-name'])['plays'].
     sum().
     reset_index().
     rename(columns = {'plays': 'total_artist_plays'})
     [['artist-name', 'total_artist_plays']]
    )

print("toplam şarkı sayimiz : ", sarki_calinmasayisi.shape[0])

#%%

kullanici_sarki_info_new = kullanici_sarki_info.merge(sarki_calinmasayisi, left_on = 'artist-name', right_on = 'artist-name', how = 'left')


#%% threshold degerinin belirlenmesi

print(sarki_calinmasayisi['total_artist_plays'].describe(include=None, exclude=None))

print(sarki_calinmasayisi['total_artist_plays'].quantile(np.arange(.1, 1, .1)))

print(sarki_calinmasayisi['total_artist_plays'].quantile(np.arange(.9, 1, .01)))


# %% threshold degerini belirledikten sonra bu degere gore veriyi ayırıyoruz

threshold = 43541
kullanici_sarki_info_populer = kullanici_sarki_info_new.query('total_artist_plays >= @threshold')

tekraronleme(kullanici_sarki_info_populer,['users', 'artist-name'])



# %%

combined = kullanici_sarki_info_populer.merge(kullanici_info, left_on = 'users', right_on = 'users', how = 'left')

ulkeler = combined.country.unique()
x=0

while (x==0):
    ulke = input("Onerıler icin ulke secimi yapin : ")
    
    if ulke in ulkeler:
        x=1
    
    else:
        print("Secilen ulke gecersiz lutfen baska bir ulke secimi yapin")
        


ulkeverisi = combined.query('country == \'{}\''.format(ulke))


if not ulkeverisi[ulkeverisi.duplicated(['users', 'artist-name'])].empty:
    initial_rows = ulkeverisi.shape[0]
    ulkeverisi = ulkeverisi.drop_duplicates(['users', 'artist-name'])
    current_rows = ulkeverisi.shape[0]
        

# %%

# sarkıcıların satır , kullanıcıların sutun , degerlerin ise calınmaları temsil ettigi matrix
ulke_verisi_matrix = ulkeverisi.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
ulke_verisi_matrix_zero_one = ulke_verisi_matrix.apply(np.sign)

# kullanıcı ve sarkıcı matrisini scipy sparse matrixine ceviriyoruz
ulke_verisi_matrix_sparse = csr_matrix(ulke_verisi_matrix.values)
ulke_verisi_matrix_zero_one_sparse = csr_matrix(ulke_verisi_matrix_zero_one.values)

#%%


# algorithm=auto parametresi kullanılan algoritmanın fit ettiğimiz verilere göre otomatik belirlenmesini saglar
# metri=cosine itemler arasındaki iliskiyi kosinus benzerligi ile bulmamızı saglar
# n_neighbours=4 model üzerinde bir vektör çevresinde komşuluk aranacak diğer vektörlerin sayısı    
model_knn = NearestNeighbors(algorithm='auto', 
                             leaf_size=30, 
                             metric='cosine',
                             metric_params=None, 
                             n_jobs=1,
                             n_neighbors=4)

model_nn_binary = NearestNeighbors(algorithm='auto', 
                             leaf_size=30, 
                             metric='cosine',
                             metric_params=None, 
                             n_jobs=1,
                             n_neighbors=4)
    
#modellerimizi fit ediyoruz 
model_knn.fit(ulke_verisi_matrix_sparse)
model_nn_binary.fit(ulke_verisi_matrix_zero_one_sparse)    

#%%

#rastgele seçilen bir şarkı için oneriler

aranan_index = np.random.choice(ulke_verisi_matrix.shape[0])
print(aranan_index)
distances, indices = model_knn.kneighbors(((ulke_verisi_matrix.iloc[aranan_index, :].values).reshape(1, -1)), n_neighbors = 2)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print("Aradıgınız sarkı ",(matrix_binary.index[aranan_index])," icin oneriler:\n")
    else:
        print(i,":",matrix_binary.index[indices.flatten()[i]], "--", distances.flatten()[i], "benzerlik oranı ile")

        
#%%

# şarkı ve öneri seçimi

sarki = input("Lutfen oneri yapılabilmesi icin dinlediginiz bir sarkinin ismini yazin :")
k = int(input("Lutfen oneride kullanılacak k (komsu) degerini girin :"))
print("\n\n")

sarki_sorgula(sarki, ulke_verisi_matrix_zero_one, model_nn_binary, k)
print("PROGRAM SONLANDI")
print("\n\n")



        
        
        
        
        
        
        
        
        
        
    