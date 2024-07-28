import numpy as np
import sys
from matplotlib import pyplot as plt

if __name__ == '__main__':
    match sys.argv[1]:
        case 'finetuning' | '1':
            plt.rcParams.update({'font.size': 14})
            plt.plot([None, 109.79463195800781, 86.26437377929688, 82.31600952148438], 'o-', color='red', linewidth=2.2, label='Training Loss')
            plt.plot([2262.114013671875, 97.18038177490234, 71.3829574584961, 60.49992752075195], 'o-', color='blue', linewidth=2.2, label='Testing Loss')
            plt.yscale('log')
            plt.xticks(range(0, 4))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

        case 'si-RMSE' | '2':
            plt.rcParams.update({'font.size': 14})
            ds_subsets = ['all', 'living rooms', 'bedrooms']
            baseline_losses = [0.260841, 0.306137, 0.220658]
            finetuned_losses = [0.186655, 0.213926, 0.164095]
            x_axis = np.arange(len(ds_subsets))
            plt.bar(x_axis - 0.2, baseline_losses, 0.4, label='Baseline')
            plt.bar(x_axis + 0.2, finetuned_losses, 0.4, label='Finetuned')
            plt.xticks(x_axis, ds_subsets)
            plt.xlabel('Data Subsets')
            plt.ylabel('si-RMSE')
            plt.legend()
        case 'si-MSE' | '3':
            plt.rcParams.update({'font.size': 14})
            ds_subsets = ['all', 'living rooms', 'bedrooms']
            baseline_losses = [0.187653, 0.258673, 0.131428]
            finetuned_losses = [0.119142, 0.161879, 0.107733]
            x_axis = np.arange(len(ds_subsets))
            plt.bar(x_axis - 0.2, baseline_losses, 0.4, label='Baseline')
            plt.bar(x_axis + 0.2, finetuned_losses, 0.4, label='Finetuned')
            plt.xticks(x_axis, ds_subsets)
            plt.xlabel('Data Subsets')
            plt.ylabel('si-MSE')
            plt.legend()

    plt.show()


