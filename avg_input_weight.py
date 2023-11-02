import os
import csv
import numpy as np
import matplotlib.pyplot as plt

ALGO_WEIGHT_MIN = -1
ALGO_WEIGHT_MAX = 1

FILES = [
    'weightConv_0_.csv',
    'inputConv_0_.csv',
    'weightConv3x3_1_.csv',
    'inputConv3x3_1_.csv',
    'weightConv3x3_2_.csv',
    'inputConv3x3_2_.csv',
    'weightConv3x3_3_.csv',
    'inputConv3x3_3_.csv',
    'weightConv3x3_4_.csv',
    'inputConv3x3_4_.csv',
    'weightConv3x3_6_.csv',
    'inputConv3x3_6_.csv',
    'weightConv3x3_7_.csv',
    'inputConv3x3_7_.csv',
    'weightConv1x1_5_.csv',
    'inputConv1x1_5_.csv',
    'weightConv3x3_8_.csv',
    'inputConv3x3_8_.csv',
    'weightConv3x3_9_.csv',
    'inputConv3x3_9_.csv',
    'weightConv3x3_11_.csv',
    'inputConv3x3_11_.csv',
    'weightConv3x3_12_.csv',
    'inputConv3x3_12_.csv',
    'weightConv1x1_10_.csv',
    'inputConv1x1_10_.csv',
    'weightConv3x3_13_.csv',
    'inputConv3x3_13_.csv',
    'weightConv3x3_14_.csv',
    'inputConv3x3_14_.csv',
    'weightConv3x3_16_.csv',
    'inputConv3x3_16_.csv',
    'weightConv3x3_17_.csv',
    'inputConv3x3_17_.csv',
    'weightConv1x1_15_.csv',
    'inputConv1x1_15_.csv',
    'weightConv3x3_18_.csv',
    'inputConv3x3_18_.csv',
    'weightConv3x3_19_.csv',
    'inputConv3x3_19_.csv',
    'weightFC_20_.csv',
    'inputFC_20_.csv',
]


def parse_csv_file(file_path):
    """
    Given a file path, parse all numbers in the CSV file and return them as a
    list of floats.
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            for num in row:
                data.append(float(num))

    d = np.array(data)
    if np.min(d) < 0:    # This math works for NeuroSim default values of algo
        d = (d + 1) / 2  # weight range of [-1, 1]
    return data, np.mean(d)


def get_averages_in_directory(dir_path):
    avg_dict = {}

    for filename in FILES:
        if filename.endswith(".csv"):
            data, average = parse_csv_file(os.path.join(dir_path, filename))
            avg_dict[filename] = (average)
            print('.', end='', flush=True)
    print()
    return avg_dict


if __name__ == "__main__":
    # Define the directory path
    dir_path = "Inference_pytorch/layer_record_ResNet18"

    # Generate histograms for all CSV files in the directory
    avg_dict = get_averages_in_directory(dir_path)

    i = 0
    print('INPUT_AVERAGE_VALUES = {')
    for key in avg_dict.keys():
        if 'weight' in key:
            continue
        print(f'\t{i}: {avg_dict[key]},')
        i += 1
    print('}')
    
    i = 0
    print('WEIGHT_AVERAGE_VALUES = {')
    for key in avg_dict.keys():
        if 'input' in key:
            continue
        print(f'\t{i}: {avg_dict[key]},')
        i += 1
    print('}')
