import pandas as pd
from sklearn.model_selection import KFold 

if __name__ == '__main__':
    df_train_vit = pd.read_csv('src/data/Fold_data/data_path_all_vit.csv')
    df_train_efficient = pd.read_csv('src/data/Fold_data/data_path_all_efficient.csv')
    kf = KFold(n_splits=10)
    i = 0
    for train_index, dev_index in kf.split(df_train_vit):
        train_content=[]
        train_name = 'src/data/Fold_data/vit_10/data_path_vit_train_{}.csv'.format(i)
        dev_name = 'src/data/Fold_data/vit_10/data_path_vit_val_{}.csv'.format(i)
        tmp_train_df = df_train_vit.iloc[train_index]
        tmp_dev_df = df_train_vit.iloc[dev_index]

        tmp_train_df.to_csv(train_name, index=0)
        tmp_dev_df.to_csv(dev_name, index=0)
        i+=1
    j = 0
    for train_index, dev_index in kf.split(df_train_efficient):
        train_content=[]
        train_name = 'src/data/Fold_data/efficient_10/data_path_efficient_train_{}.csv'.format(j)
        dev_name = 'src/data/Fold_data/efficient_10/data_path_efficient_val_{}.csv'.format(j)
        tmp_train_df = df_train_efficient.iloc[train_index]
        tmp_dev_df = df_train_efficient.iloc[dev_index]

        tmp_train_df.to_csv(train_name, index=0)
        tmp_dev_df.to_csv(dev_name, index=0)
        j+=1