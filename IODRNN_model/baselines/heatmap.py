import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = r"F:\本科\Laptop\spatial_data\area.csv"

if __name__ == "__main__":
    density = np.load("density.npy", allow_pickle=True)
    print(density)
    # density = np.around(((density - np.min(density)) / (np.max(density) - np.min(density))) * 10,0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(density)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(10):
        for j in range(10):
            text = ax.text(i, j, density[i, j],
                           ha="center", va="center", color="w")
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    ax.set_title("GPS Density in Selected Area")
    plt.colorbar(im)
    # plt.tight_layout()
    # plt.savefig('heatmat.svg', format='svg')
    plt.show()