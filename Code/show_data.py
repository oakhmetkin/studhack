from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

from dataloader import import_dataset_from_file


fig, axs = plt.subplots(2, 3)
axs = axs.flatten()

for j in range(5):
    # loading dataset
    print(f'Loading dataset #{j+1}')
    df = import_dataset_from_file(f'../Data/Map_{j+1}.txt')
    XY, z = df[['x', 'y']].values, df['z'].values

    # sizes
    xmin, ymin, xmax, ymax = XY[:, 0].min(), XY[:, 1].min(), XY[:, 0].max(), XY[:, 1].max()

    # knn training
    print('kNN training')
    knn = KNeighborsRegressor(n_neighbors=4, weights='distance', metric='l2')
    knn.fit(XY, z)

    # showing
    h, w = (1000, 1000) # height X width
    img = np.zeros((h, w))
    y_range = np.linspace(ymin, ymax, h)

    for i in range(h):
        y = y_range[i]
        xy = np.array([(x, y) for x in np.linspace(xmin, xmax, w)])
        pred = knn.predict(xy)
        img[i, :] = pred

    axs[j].set_title(f'Map {j+1}')
    axj = axs[j].imshow(img, cmap='gist_rainbow')

plt.show()
