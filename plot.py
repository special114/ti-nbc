import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


index = {}


def extract_info(line, num_of_cpus):
    start_idx = line.index('model') + len('model ')
    stop_idx = line.index('...')
    model_name = line[start_idx:stop_idx]
    start_idx = line.index('OK in') + len('OK in ')
    stop_idx = line.index('s]')
    time = float(line[start_idx:stop_idx])
    if model_name not in index:
        index[model_name] = [time]
    else:
        index[model_name].append(time)


def read_file(num_of_cpus):
    file_name = f"data/out_{num_of_cpus}_cpu.txt"
    with open(file_name, 'r') as file:
        for line in file.readlines():
            if "OK in" in line:
                extract_info(line, num_of_cpus)
    file.close()


def pr_list(new_list, t):
    species = [z[0] for z in new_list]
    penguin_means = {}
    for i in range(0, 5):
        penguin_means[i + 1] = [z[1][i] for z in new_list]

    x = np.arange(len(species))  # the label locations
    width = 0.13  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Elapsed time (s)')
    ax.set_xticks(x + width, species, rotation='vertical')
    ax.legend(loc='upper left', ncols=3, title='Num of executors')

    fig.savefig(f'output{t}.png')


def process(list):
    new_list = list[0:10]
    pr_list(new_list, 1)
    new_list = list[10:20]
    pr_list(new_list, 2)
    new_list = list[20:30]
    pr_list(new_list, 3)
    new_list = list[30:40]
    pr_list(new_list, 4)
    new_list = list[40:43]
    pr_list(new_list, 5)




def zzzzz():
    species = [1, 2, 3, 4, 5]
    y = [669.74, 435.64, 395.89, 369.38, 398.03]

    fig, ax = plt.subplots(layout='constrained')

    for i, x in enumerate(species):
        res = ax.bar(x, y[i])
        ax.bar_label(res, padding=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Execution time (s)')
    ax.set_xlabel('Num of executors')
    ax.set_title('Total execution time for number of executors')

    fig.savefig(f'overall.png')


def main():
    for num_of_cpu in range(1, 6):
        read_file(num_of_cpu)
    s = sorted(index.items(), key=lambda x: max(x[1]))
    print(s)

    d = {
        'model': [z[0] for z in s],
        '1_executor': [z[1][0] for z in s],
        '2_executors': [z[1][1] for z in s],
        '3_executors': [z[1][2] for z in s],
        '4_executors': [z[1][3] for z in s],
        '5_executors': [z[1][4] for z in s],
    }
    print(pd.DataFrame(data=d).to_string())

    process(s)
    zzzzz()


main()
