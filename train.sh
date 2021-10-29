#!/bin/bash
TYPE=train
VIDEO_PATH="../../algo-2021/dataset/videos/video_5k/train_5k/"
if [ -z "$1" ]; then
    echo "[Warning] TYPE is not set, using 'train' as default"
else
    TYPE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    echo "[Info] TYPE is ${TYPE}"
fi

if [ -z "$2" ]; then
    echo "[Warning] VIDEO_PATH is not set, using '../../algo-2021/dataset/videos/video_5k/train_5k' as default"
else
    VIDEO_PATH=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    echo "[Info] VIDEO_PATH is ${VIDEO_PATH}"
fi


if [ "$TYPE" = "frame" ]; then
    files=$(ls $VIDEO_PATH)
    train_frame_dir="../dataset/train/frame/"
    image_name='/mp4-%05d.jpeg'
    if [ ! -d "$train_frame_dir" ];then
        mkdir -p $train_frame_dir
        echo "创建帧存放文件夹成功"
    else
        echo "帧存放文件夹已经存在"
    fi
    for file in $files
    do
        single_frame_dir=$train_frame_dir${file%.*}
        video_file=$VIDEO_PATH${file}
        frame_name=$single_frame_dir$image_name
        if [ ! -d "$single_frame_dir" ];then
            mkdir $single_frame_dir
        fi
        ffmpeg -i ${video_file} -f image2 -vf fps=fps=1 -qscale:v 2 ${frame_name}
    done
exit 0
elif [ "$TYPE" = "extract" ]; then
  vit_csv_file="pre/video_feature_extract/csv_file/input_train_vit.csv"
  efficient_csv_file="pre/video_feature_extract/csv_file/input_train_efficient.csv"
  if [ ! -d "$vit_csv_file" ];then
      time python pre/video_feature_extract/csv_gen_train.py
  fi
  echo "[Info] Start extract video features with vit"
  time python pre/video_feature_extract/extract_vit.py --csv "${vit_csv_file}"
  echo "[Info] Start extract video features with efficient"
  time python pre/video_feature_extract/extract_efficient.py --csv "${efficient_csv_file}"
  exit 0

elif [ "$TYPE" = "extract_vit" ]; then
  vit_csv_file="pre/video_feature_extract/csv_file/input_train_vit.csv"
  if [ ! -d "$single_frame_dir" ];then
      time python pre/video_feature_extract/csv_gen_train.py
  fi
  echo "[Info] Start extract video features with vit"
  time python pre/video_feature_extract/extract_vit.py --csv "${vit_csv_file}"
  exit 0

elif [ "$TYPE" = "train" ]; then
  echo "[Info] Start train model"
  time python train.py
  exit 0
  

elif [ "$TYPE" = "generate" ]; then
  echo "[Info] Generate train Dataset"
  time python src/data/data_path_gen_efficient.py
  time python src/data/data_path_gen_vit.py
  time python src/data/fold_data_gen.py
  exit 0
fi




