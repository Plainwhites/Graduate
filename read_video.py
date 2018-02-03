import cv2
import os
import numpy as np
import tensorflow as tf


data_path = '/home/zy/Data/Florence_3d_actions/'


def count_video_frames(path):
    """Count the shortest and longest video"""
    # shortest = 100
    # longest = 0
    write_path = '/home/zy/Data/count.txt'
    f = open(write_path, 'w')
    videonames = os.listdir(path)
    for each_name in videonames:
        count = 0
        videoname = path + each_name
        video = cv2.VideoCapture(videoname)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            count += 1

        f.write('%s %d\n' % (each_name, count))

        # shortest = count if count < shortest else shortest
        # longest = count if count > longest else longest

    # return shortest, longest


# if __name__ == '__main__':
#     count_video_frames(data_path)


def get_list_of_frames(video_name):
    """:param
        video_name: video name
        return: video frames and num_frame
    """
    count = 0
    ret = []
    video = cv2.VideoCapture(video_name)
    while True:
        flag, frame = video.read()
        if not flag:
            break
        count += 1
        ret.append(frame)
    return ret, count


def get_random_video_frames(data_path):
    videonames = os.listdir(data_path)
    serial_number = list(np.random.randint(low=0, high=215, size=10))
    ret = []
    ret_label = []
    num_frame = []
    for i in serial_number:
        video, count = get_list_of_frames(data_path+videonames[i])
        ret.append(video)
        ret_label.append(float(videonames[i][-5:-4]))
        num_frame.append(count)
    label = tf.one_hot(indices=ret_label, depth=9)
    return ret, label, num_frame

# if __name__ == '__main__':
#     # videopath = '/home/zy/Data/Florence_3d_actions/GestureRecording_Id1actor1idAction1category1.avi'
#     frames = get_random_video_frames(data_path)
#     print(frames)









