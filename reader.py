import os
import numpy as np


SKELETON_JOINT_NUMBER = 15


def get_file_name(folder):
    """
    :param folder: the folder of the file
    :return: a list of file_name
    """
    file_name = os.listdir(folder)
    file_name.sort()
    return file_name


def read_file(path, max_x, min_x, max_y, min_y, max_z, min_z):
    """
    :param path: the full path concatenated with the filename
           max_coor: find the max and min coordinate to normalize the skeleton
           min_coor: find the max and min coordinate to normalize the skeleton
    :return: a list the file context
    """
    file = []
    with open(path) as f:
        all_lines = f.readlines()
        for each_line in all_lines:
            line = list(each_line.split())
            for i in range(len(line)):
                line[i] = float(line[i])
                if i % 3 == 0:
                    max_x = line[i] if line[i] > max_x else max_x
                    min_x = line[i] if line[i] < min_x else min_x
                if i % 3 == 1:
                    max_y = line[i] if line[i] > max_y else max_y
                    min_y = line[i] if line[i] < min_y else min_y
                if i % 3 == 2:
                    max_z = line[i] if line[i] > max_z else max_z
                    min_z = line[i] if line[i] < min_z else min_z
            file.append(line)

    return file, max_x, min_x, max_y, min_y, max_z, min_z


def read_all_file(folder):
    """
    :param folder: the file folder
    :return: a normalized list of skeleton joint coordinates
    """

    files = []
    # min_z = 1.0 because min z is bigger than 0.0
    max_x, min_x, max_y, min_y, max_z, min_z = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    file_name = get_file_name(folder=folder)
    for each_file in file_name:
        file, max_x, min_x, max_y, min_y, max_z, min_z = read_file(path=folder+each_file,
                                                                   max_x=max_x,
                                                                   min_x=min_x,
                                                                   max_y=max_y,
                                                                   min_y=min_y,
                                                                   max_z=max_z,
                                                                   min_z=min_z)
        files.append(file)
    div_x = max_x - min_x
    div_y = max_y - min_y
    div_z = max_z - min_z

    for i in range(len(files)):
        for j in range(len(files[0])):
            for k in range(3*SKELETON_JOINT_NUMBER):
                if k % 3 == 0:
                    files[i][j][k] = (files[i][j][k] - min_x) / div_x
                    continue
                if k % 3 == 1:
                    files[i][j][k] = (files[i][j][k] - min_y) / div_y
                    continue
                if k % 3 == 2:
                    files[i][j][k] = (files[i][j][k] - min_z) / div_z

    return files


def random_input(data):
    # generate the random train batch
    # [batch_size, step_num, elem_size]
    serial_number = list(np.random.randint(low=0, high=215, size=10))
    ret = []
    for _ in serial_number:
        ret.append(data[_])

    return ret
