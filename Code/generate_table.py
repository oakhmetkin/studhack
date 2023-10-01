from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np

from Code.main import import_dataset_from_file


knns = []

for i in range(5):
    print(f'Loading dataset #{i+1}')
    df = import_dataset_from_file(f'../Data/Map_{i+1}.txt')
    X, y = df[['x', 'y']].values, df['z'].values

    print('kNN training')
    knn = KNeighborsRegressor(n_neighbors=4, weights='distance', metric='l2')
    knn.fit(X, y)
    knns.append(knn)


points_df = import_dataset_from_file('../Data/Point_dataset.txt')

cols = ['x', 'y', 'map1', 'map2', 'map3', 'map4', 'map5', 'F']
df = pd.DataFrame(columns=cols)
df[['x', 'y', 'F']] = points_df[['x', 'y', 'z']]

for i, knn in enumerate(knns):
    xy = df[['x', 'y']].values.astype(np.float32)
    df[f'map{i+1}'] = knn.predict(xy)

df.to_csv('../Data/new_dataset.csv')
