from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np

from dataloader import import_dataset_from_file


knns = []

for i in range(5):
    print(f'Loading dataset #{i+1}')
    df = import_dataset_from_file(f'../Data/Map_{i+1}.txt')
    X, y = df[['x', 'y']].values, df['z'].values

    print('kNN training')
    knn = KNeighborsRegressor(n_neighbors=4, weights='distance', metric='l2')
    knn.fit(X, y)
    knns.append(knn)


