import numpy as np
import matplotlib.pyplot as plt

from dataloader import import_dataset_from_file


maps = []
images = []

for i in range(1):
    df = import_dataset_from_file(f'../Data/Map_{i+1}.txt')
    maps.append(df)

    unique_y = df['y'].unique()
    image = []

    for y in unique_y:
        row = df[df['y'] == y]['z']
        image.append(row)
        print(row.shape)
    
    images.append(np.array(image))

# print([img.shape for img in images])


