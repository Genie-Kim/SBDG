import argparse
from domainbed import datasets
import pandas as pd
import os
import datetime
import torch
from torchvision.datasets import ImageFolder

"""
function about imbalance datasets with pandas dataframe
the dataframe's default shape is
dom cls img
2   133 imgpath
...
"""


def split_tetr_domclstable(cls_sampled_table, num_testimgs): # return (test_list,train_list)
    pivot_dict_samenum = domclsdf_to_pivotdict_samenum(cls_sampled_table, num_testimgs)

    return sampling_from_pivotdict(pivot_dict_samenum ,cls_sampled_table)

def domclsdf_to_pivotdict_samenum(dom_cls_img_df, samenum):
    # dom_cls_img_df columns = dom, cls, image path
    dom_list = dom_cls_img_df.groupby(by='dom').count().index.to_list()
    cls_list = dom_cls_img_df.groupby(by='cls').count().index.to_list()
    return {dom :{cls :samenum for cls in cls_list} for dom in dom_list}


def pivotdf_to_dict(pivot_df):
    # pivot data frame shape : row name cls, column name domain, value count
    return {k[1] :v for k ,v in pivot_df.to_dict().items()}

def sampling_from_pivotdict(pivot_dict ,dom_cls_img_df):
    # make dom_cls_img_df to meet pivot_dict number of images by random sampling
    # pivot_dict is {dom:{cls:num,,,},,,}
    # dom_cls_img_df columns = dom, cls, image path
    sampled_list =[]
    remained_list =[]
    for dom ,v in pivot_dict.items():
        for cls ,num in v.items():
            temp_df = dom_cls_img_df.loc[
                (dom_cls_img_df['dom'] == dom) & (dom_cls_img_df['cls'] == cls)]
            sample_df = temp_df.sample(n=num)
            remain_df = temp_df.drop(sample_df.index)

            sampled_list = sampled_list + sample_df.values.tolist()
            remained_list = remained_list + remain_df.values.tolist()

    return (sampled_list, remained_list)


def remaincls_numimgs_df(dom_cls_table, num_cls):
    count_table = dom_cls_table.pivot_table(index='cls', columns='dom', aggfunc='count')  # image count pivot table
    newdf = count_table
    while (len(newdf) > num_cls):
        newdf.min().idxmin()
        minimum_class = newdf.loc[:, newdf.min().idxmin()].idxmin()
        newdf = newdf.drop(minimum_class, axis=0)
    return (newdf.index.tolist(), newdf.min().min())  #


def csv_to_imbalance_csv(train_df, class_or_domain,imbalance_rate,minor_domain,target_doms):
    # minor_domain is number
    # target doms is number of list
    # 1 <  imbalance_rate
    dom_list = train_df.groupby(by='dom').count().index.to_list()
    dom_to_idx = {k:v for v,k in enumerate(dom_list)}
    source_doms = [x for x in range(len(dom_list)) if x not in target_doms] # source domain numbering.
    # make imbalance to train dataset, minor domain must be one int
    if class_or_domain == 'domain':
        # make domain imbalance
        if minor_domain not in list(range(len(dom_list))):
            assert False, 'minor domain selection is wrong number'
        elif minor_domain in target_doms:
            assert False, 'minor domain is target domain!!'

        train_img_maxnum = int(train_df.groupby(by=['dom','cls']).count().values[0])

        # assume number of minor classes is one
        Ds = len(source_doms)  # number of source domains
        # balance dataset number.
        B = int(((Ds - 1) * train_img_maxnum) / Ds)  # To make the number of images(which are network show) independent with imbalance rate.(assume imbalance rate is infinite)
        num_minor = int(B * Ds / (1 + (Ds - 1) * imbalance_rate))  # image number of minor domain of imbalance
        num_major = num_minor * imbalance_rate  # image number of major domain of imbalance

        temp_pivotdict = domclsdf_to_pivotdict_samenum(train_df,num_major)  # set all dom cls number to num major.
        for dom, v in temp_pivotdict.items():
            for cls, num in v.items():
                if dom_to_idx[dom] in target_doms:
                    temp_pivotdict[dom][cls] = train_img_maxnum  # set target(not change from traindf)
                elif dom_to_idx[dom] == minor_domain:
                    temp_pivotdict[dom][cls] = num_minor  # set minor domain
        imb_list, _ = sampling_from_pivotdict(temp_pivotdict, train_df)

        imb_df = pd.DataFrame(imb_list, columns=['dom', 'cls', 'img'])
        print('####imbalance dataset count table of imbalance dataset######')
        print(imb_df.pivot_table(index='cls', columns='dom', aggfunc='count'))
        print('############################################################')
        return imb_df


    elif class_or_domain == 'class':
        # make class imbalance
        print(1)

    else:
        assert False, 'there is no description for class or domain imbalance'

def mk_imbalance_newdatacsv(dom_dataset_list, domain_names,test_rate, cls_num):


    temp_list = []
    for domain, dataset in enumerate(dom_dataset_list):
        for img, cls in dataset.imgs:
            temp_list.append([domain, cls, img])

    # make domain class imagepath column dataframe of pandas.
    dom_cls_table = pd.DataFrame(temp_list, columns=['dom', 'cls', 'img'])

    # get maximum number of images table which correct the num_cls number.
    remain_cls_list, img_num = remaincls_numimgs_df(dom_cls_table, cls_num)

    # sampling by remain class.
    cls_sampled_table = dom_cls_table[dom_cls_table['cls'].apply(lambda x: True if x in remain_cls_list else False)]
    sampled_list, _ = split_tetr_domclstable(cls_sampled_table,
                                                 img_num)  # Match the number of images of domain and class pair equally.
    cls_sampled_table = pd.DataFrame(sampled_list, columns=['dom', 'cls', 'img'])

    # split test & (training and val) set
    num_testimgs = int(img_num * test_rate)  # number of test images per (domain, class)
    # Do not reset index at cls_sampled_table, before this line.
    testlist, trainlist = split_tetr_domclstable(cls_sampled_table, num_testimgs)

    train_df = pd.DataFrame(trainlist, columns=['dom', 'cls', 'img'])
    test_df = pd.DataFrame(testlist, columns=['dom', 'cls', 'img'])  # do not use on training set

    # save csv test set of imbalance data to dataset path
    named_test_df = replace_domcls_toname(test_df, domain_names, dom_dataset_list[0].class_to_idx)
    print('test dataset count table of imbalance dataset')
    named_test_df = named_test_df.pivot_table(index='cls', columns='dom', aggfunc='count')
    print(named_test_df.rename(columns={i: name for i, name in enumerate(domain_names)},
                               index={v: k for k, v in dom_dataset_list[0].class_to_idx.items()}))

    # save csv test set of imbalance data to dataset path
    named_train_df = replace_domcls_toname(train_df, domain_names, dom_dataset_list[0].class_to_idx)
    print('train dataset count table of imbalance dataset')
    named_train_df = named_train_df.pivot_table(index='cls', columns='dom', aggfunc='count')
    print(named_train_df.rename(columns={i: name for i, name in enumerate(domain_names)},
                               index={v: k for k, v in dom_dataset_list[0].class_to_idx.items()}))

    return (train_df,test_df)


def replace_domcls_toname(df, domain_names, class_to_idx):

    df['dom'] = df['dom'].replace(list(range(len(domain_names))), domain_names)
    df['cls'] = df['cls'].replace([v for k, v in class_to_idx.items()], class_to_idx.keys())

    return df


def dataset_path(data_root,dataset):
    if dataset == 'DomainNet':
        datapath = os.path.join(data_root,'domain_net/')
        if not os.path.isdir(datapath):
            assert False, 'there is no dataset'
        return datapath
    else:
        assert False, 'no dataset in dataset path function.'


def datapath_to_domaindatasetlist(datapath):
    domain_list = []
    for item in os.listdir(datapath):
        if os.path.isdir(os.path.join(datapath, item)):
            domain_list.append(item)
    domain_list.sort()

    domain_dataset_list=[]
    for domain in domain_list:
        domain_dataset = ImageFolder(os.path.join(datapath, domain))
        domain_dataset_list.append(domain_dataset)

    return (domain_dataset_list, domain_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make a imbalance dataset csv')
    parser.add_argument('--command', choices=['totalnew', 'anotherimb']) # totalnew is making new random dataset csv(include test dataset)
    parser.add_argument('--imb_rate', type=int, default=5)
    parser.add_argument('--cls_num', type=int, default=3)
    parser.add_argument('--test_rate', type=float, default=0.2)
    parser.add_argument('--minor_domain', type=int, default=3)
    parser.add_argument('--target_doms', type=int, nargs='+', default=[0])  # 도메인 이름 알파벳으로 소팅하고 난뒤 index로 타겟 도메인 지
    parser.add_argument('--output_dir', type=str, default="../imbalance_result_output")
    parser.add_argument('--dataroot_dir', type=str,default="/home/genie/2hddb/dg_dataset")
    parser.add_argument('--train_csv', type=str,default="/home/genie/PycharmProjects/DomainBed/domainbed/imbalance_result_output/20201107180355_dataset_DomainNet_numcls_3_testrate_02/train.csv")
    parser.add_argument('--dataset', choices=datasets.DATASETS)
    parser.add_argument('--imb_type', choices=['class','domain']) # 'class_or_domain' -> imb_type
    args = parser.parse_args()



    if args.command == 'totalnew':
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        folder_name = '_'.join([now,'dataset',args.dataset,'numcls',str(args.cls_num),'testrate',str(args.test_rate).replace('.','')])

        folder_path = os.path.join(args.output_dir,folder_name)
        os.makedirs(folder_path,exist_ok=True)

        datapath = dataset_path(args.dataroot_dir,args.dataset)
        dom_dataset_list, domain_list = datapath_to_domaindatasetlist(datapath)

        train_df, test_df = mk_imbalance_newdatacsv(dom_dataset_list,domain_list,args.test_rate, args.cls_num)
        train_df.to_csv(os.path.join(folder_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(folder_path, 'test.csv'), index=False)


    elif args.command == 'anotherimb': # make imbalance data from train csv
        folder_name = os.path.dirname(args.train_csv)
        folder_path = os.path.join(args.output_dir,folder_name)
        train_df = pd.read_csv(args.train_csv)
        imb_df = csv_to_imbalance_csv(train_df, args.imb_type, args.imb_rate, args.minor_domain, args.target_doms)

        targets=''.join([str(x) for x in args.target_doms])
        imb_df.to_csv(os.path.join(folder_path, 'imb_targets_'+targets+'_minor_'+str(args.minor_domain) +'_imbrate_'+str(args.imb_rate)+'_clsordom_'+args.imb_type + '.csv'), index=False)