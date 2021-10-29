import csv
import os


with open("pre/video_feature_extract/csv_file/input_train_vit.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["video_path","feature_path"])
    txt_root_dir = r'../dataset/train/frame/'
    video_dir = '../dataset/train/video_npy/vit/'
    file_list = os.listdir(txt_root_dir)
    file_list.sort()
    for line in file_list:
        ord_path = txt_root_dir + '/' + line
        name = line.split('.')[0]
        new_path = video_dir + name + '.npy'
        writer.writerow([ord_path,new_path])
with open("pre/video_feature_extract/csv_file/input_train_efficient.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["video_path","feature_path"])
    txt_root_dir = r'../dataset/train/frame/'
    video_dir = '../dataset/train/video_npy/efficient/'
    file_list = os.listdir(txt_root_dir)
    file_list.sort()
    for line in file_list:
        ord_path = txt_root_dir + '/' + line
        name = line.split('.')[0]
        new_path = video_dir + name + '.npy'
        writer.writerow([ord_path,new_path])
            