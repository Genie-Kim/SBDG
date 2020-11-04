import os
import pandas as pd
import random
import shutil


def mk_imbalance_datalist(img_dataset_path,csv_pandas,target_domain):
    # img_dataset_path/domain/class/ 안에 이미지들이 있는 구조.
    # source_domain to be imbalanced dataset.
    # dom_cls_sampled_path에는 target doamin과 sampling된 source domain의 이미지들이 들어있다.

    source_domain = csv_pandas.index.to_list()
    classes = csv_pandas.columns.to_list()
    domains = source_domain+target_domain
    dom_cls_comb_list = [(x, i) for x in domains for i in classes]
    dom_cls_sampled_path = dict.fromkeys(dom_cls_comb_list)


    for domain in domains:
        for cls in classes:
            path_D_C = os.path.join(img_dataset_path,domain,cls) # img_dataset_path/domain/class/ 안에 이미지들이 있는 구조.
            imgpath = []
            for imgfile in os.listdir(path_D_C):
                temp = os.path.join(path_D_C, imgfile)
                if os.path.isfile(temp):
                    imgpath.append(temp)
                else:
                    print('there is item which is not file in',temp)

            if domain in source_domain:
                dom_cls_sample_limit = int(csv_pandas.loc[domain, cls])
                if len(imgpath) < dom_cls_sample_limit:
                    print(domain +', '+ cls + ' data sample is Not enough, So oversampling..')
                sampled_imgpath = random.choices(imgpath,k=dom_cls_sample_limit)
            else:
                sampled_imgpath = imgpath

            dom_cls_sampled_path[(domain, cls)] = sampled_imgpath

    for key,val in dom_cls_sampled_path.items():
        print(key[0], key[1], len(val))

    return dom_cls_sampled_path


def copy_struct_bysampled_data(dom_cls_sampled_path, save_copy_path, csv_pandas):
    # dom_cls_sampled_path : key = (domain,class), value = [sampled img path1,,,,,]
    # This function save the sampled image to save_copy_path/domain/class
    # source target infor saved in info_domcls_imbalance.csv
    try:
        os.mkdir(save_copy_path)
    except FileExistsError:
        print('there is already exist directory : ',save_copy_path)
        os.mkdir(save_copy_path+'_new')

    csv_save_path = os.path.join(save_copy_path,'info_domcls_imbalance.csv')
    csv_pandas.loc['Column_Total'] = csv_pandas.sum(numeric_only=True, axis=0)
    csv_pandas.loc[:, 'Row_Total'] = csv_pandas.sum(numeric_only=True, axis=1)

    csv_pandas.to_csv(csv_save_path)

    for key, val in dom_cls_sampled_path.items():
        save_dom_cls_path=os.path.join(save_copy_path,key[0],key[1])
        os.makedirs(save_dom_cls_path)
        for source_img_path in val:
            temp = os.path.join(save_dom_cls_path,os.path.basename(source_img_path))
            shutil.copy(source_img_path,temp)


if __name__ == '__main__':

    img_dataset_path = os.path.expanduser('~/pacs_dataset/Raw images/kfold/')
    imbalance_csv_path = os.path.expanduser('~/pacs_dataset/balanced.csv')
    split_idx = 3 # domain number
    saving_path = os.path.expanduser('~/pacs_dataset/Raw images/balanced')

    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    # classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    target_domain = [domains.pop(split_idx)]
    source_domain = domains

    csv_pandas = pd.read_csv(imbalance_csv_path, index_col=0) # column index = source domain, row index = classes

    dom_cls_sampled_path = mk_imbalance_datalist(img_dataset_path,csv_pandas,target_domain)
    copy_struct_bysampled_data(dom_cls_sampled_path, saving_path,csv_pandas)




