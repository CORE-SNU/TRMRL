import os
import argparse
import numpy as np
import pandas
import matplotlib.pyplot as plt


def main(csv_dir_name):
    res_dir = os.path.join(csv_dir_name, 'plots')
    os.makedirs(res_dir, exist_ok=True)
    fontsize = 18

    csv_file_path = os.path.join(csv_dir_name, 'progress.csv')
    data = pandas.read_csv(csv_file_path, header=0)

    columns = data.columns

    #  mean, std, max, min
    columns_with_stat = ['Log Pis', 'Policy log std', 'Policy mu', 'Q Predictions', 'V Predictions']
    columns_not_needed = ['Epoch', 'Epoch Time (s)', 'Number of env steps total', 'Number of rollouts total', 'Number of train steps total', 'Sample Time (s)',
                          'Total Train Time (s)', 'Train Time (s)']

    columns_without_stat = []
    for col2 in columns:
        if col2 not in columns_not_needed:
            flag = False

            for col in columns_with_stat:
                if col2.startswith(col):
                    flag = True
                    break
            if not flag:
                columns_without_stat.append(col2)

    for col in columns_without_stat:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(data[col])
        ax.set_xlim(0., len(data))
        ax.set_xlabel('iteration', fontsize=fontsize)
        ax.set_title(col, fontsize=fontsize)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(res_dir, '{}.png'.format(col)), dpi=60)
        plt.clf(), plt.cla()
        plt.close(fig)

    for col in columns_with_stat:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        y_mean = data[col + ' Mean']
        y1 = y_mean - data[col + ' Std']
        y2 =y_mean + data[col + ' Std']

        y_min = data[col + ' Min']
        y_max = data[col + ' Max']

        x = np.arange(len(data))

        ax.plot(x, y_mean, color='tab:blue')
        ax.plot(x, y_min, linestyle='dotted', color='tab:blue')
        ax.plot(x, y_max, linestyle='dotted', color='tab:blue')

        ax.fill_between(x, y1, y2, alpha=0.2, color='tab:blue')
        ax.set_xlim(0., len(data))
        ax.set_xlabel('iteration', fontsize=fontsize)
        ax.set_title(col, fontsize=fontsize)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(res_dir, '{}.png'.format(col)), dpi=60)
        plt.clf(), plt.cla()
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, type=str)
    args = parser.parse_args()
    main(args.dir)
