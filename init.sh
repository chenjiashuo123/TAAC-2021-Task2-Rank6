#!/bin/bash
# #################### get env directories
# CONDA_ROOT

CONDA_NEW_ENV=pytorch_py3
sudo apt-get update
sudo apt-get install -y apt-utils
sudo apt-get install -y libsndfile1-dev ffmpeg
source activate ${CONDA_NEW_ENV}

pip install -r requirement.txt



DATASET="../../algo-2021/dataset/tagging/"
if [ -z "$1" ]; then
    echo "[Warning] TYPE is not set, using '../../algo-2021/dataset/tagging/' as default"
else
    DATASET=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    echo "[Info] DATASET is ${DATASET}"
fi




dir="../dataset/train/frame"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi

dir="../dataset/train/text_txt" 
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi


dir="../dataset/train/audio_npy"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi

dir="../dataset/train/raw_video"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi

dir="../dataset/train/video_npy/vit"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi

dir="../dataset/train/video_npy/efficient"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi

# 复制文本
text_dir="tagging_dataset_train_5k/text_txt/tagging/"
text_name="tagging_dataset_train_5k/text_txt/tagging/*"
source_dir=$DATASET$text_dir
target_dir="../dataset/train/text_txt/"

if [ -d $source_dir ];then
    echo "文本数据集路径正确"
else
    echo "文本数据集路径错误"
    exit 0
fi
source_dir=$DATASET$text_name

cp $source_dir $target_dir
echo "复制文本数据成功"


# 复制音频
audio_dir="tagging_dataset_train_5k/audio_npy/Vggish/tagging/"
audio_name="tagging_dataset_train_5k/audio_npy/Vggish/tagging/*"
source_dir=$DATASET$audio_dir
target_dir="../dataset/train/audio_npy/"

if [ -d $source_dir ];then
    echo "音频数据集路径正确"
else
    echo "音频数据集路径错误"
    exit 0
fi
source_dir=$DATASET$audio_name

cp $source_dir $target_dir
echo "复制音频数据成功"


dir="../dataset/test_2nd/frame"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi

dir="../dataset/test_2nd/text_txt" 
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi


dir="../dataset/test_2nd/audio_npy"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi

dir="../dataset/test_2nd/raw_video"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi

dir="../dataset/test_2nd/video_npy/vit"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi


dir="../dataset/test_2nd/video_npy/efficient"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    echo "创建文件夹成功"
    else
    echo "文件夹已经存在"
fi

# 复制文本
text_dir="tagging_dataset_test_5k_2nd/text_txt/tagging/"
text_name="tagging_dataset_test_5k_2nd/text_txt/tagging/*"
target_dir="../dataset/test_2nd/text_txt/"
source_dir=$DATASET$text_dir
if [ -d $source_dir ];then
    echo "测试集文本数据集路径正确"
    else
    echo "测试集文本数据集路径错误"
    exit 0
fi
source_dir=$DATASET$text_name
cp $source_dir $target_dir
echo "复制测试集文本数据集成功"


# 复制音频
audio_dir="tagging_dataset_test_5k_2nd/audio_npy/Vggish/tagging/"
audio_name="tagging_dataset_test_5k_2nd/audio_npy/Vggish/tagging/*"
source_dir=$DATASET$audio_dir
target_dir="../dataset/test_2nd/audio_npy/"

if [ -d $source_dir ];then
    echo "测试集音频数据集路径正确"
    else
    echo "测试集音频数据集路径错误"
    exit 0   
fi

source_dir=$DATASET$audio_name
cp $source_dir $target_dir
echo "复制数据集成功"





