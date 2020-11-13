import argparse
from domainbed import datasets
import pandas as pd
import os
import datetime
import torch
from torchvision.datasets import ImageFolder
import re
"""
function about imbalance datasets with pandas dataframe
the dataframe's default shape is
dom cls img
2   133 imgpath
...
"""


def split_tetr_domclstable(cls_sampled_table, num_sampledimgs,target_doms=[], check_valnum_doms=[]): # return (test_list,train_list)
    pivot_dict_samenum = domclsdf_to_pivotdict_samenum(cls_sampled_table, num_sampledimgs)

    if len(check_valnum_doms)>0:
        # 도메인 클래스 별 image개수가 validation개수보다 부족하면, 그 해당 이미지 개수의 1/3만 validation set으로 한다.
        for dom in check_valnum_doms:
            domname = list(pivot_dict_samenum.keys())[dom]
            for cls, num in pivot_dict_samenum[domname].items():
                num_img = int(cls_sampled_table.groupby(by=['dom','cls']).count().loc[domname,cls])
                if num_sampledimgs > num_img:
                    print(domname, cls, ' : this settings number of images are less than validation number')
                    pivot_dict_samenum[domname][cls] = num_img/3 # validation set

    # make the number of sampled images in target domain 0. fixed target domain들은 sampling 하지 않는다.
    for dom in target_doms:
        domname = list(pivot_dict_samenum.keys())[dom]
        for cls, num in pivot_dict_samenum[domname].items():
            pivot_dict_samenum[domname][cls]=0

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
            sample_df = temp_df.sample(n=int(num))
            remain_df = temp_df.drop(sample_df.index)

            sampled_list = sampled_list + sample_df.values.tolist()
            remained_list = remained_list + remain_df.values.tolist()

    return (sampled_list, remained_list)


def remaincls_numimgs_df(dom_cls_table, num_cls,fixtarget_doms):
    newdf = dom_cls_table.copy()
    for t in fixtarget_doms: # drop the fixed target domains data for searching.
        newdf.drop(newdf[newdf['dom'] == t].index, axis=0, inplace=True)

    newdf = newdf.pivot_table(index='cls', columns='dom', aggfunc='count')  # image count pivot table

    while (len(newdf) > num_cls):
        newdf.min().idxmin()
        minimum_class = newdf.loc[:, newdf.min().idxmin()].idxmin()
        newdf = newdf.drop(minimum_class, axis=0)
    return (newdf.index.tolist(), newdf.min().min())  #


def csv_to_imbalance_csv(train_csv_path, class_or_domain,imbalance_rate,minor_domain,domain_namelist):
    # minor_domain is number
    # domain_namelist is domain idx to name.
    # 1 =<  imbalance_rate
    train_df = pd.read_csv(train_csv_path)
    source_doms = train_df.groupby(by='dom').count().index.to_list() # name domain and class.

    # make imbalance to train dataset, minor domain must be one int
    if class_or_domain == 'domain':
        # make domain imbalance
        assert (domain_namelist[minor_domain] in source_doms), 'minor domain is not in source domains'

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
                if dom == domain_namelist[minor_domain]:
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

def mk_imbalance_newdatacsv(cls_sampled_csvpath,test_rate,val_rate,fixtarget_doms,running_target_doms):
    # running_target_doms is current target which is not permanant
    # test_rate,val_rate is for running source domains(in code, img_num)
    # cls_sampled_table is changed domain number to domain names
    for t in running_target_doms:
        assert t not in fixtarget_doms, 'running target dom is in the fixtarget_doms'

    cls_sampled_table = pd.read_csv(cls_sampled_csvpath)

    img_num = int(os.path.splitext(os.path.basename(cls_sampled_csvpath))[0].split('_')[-1]) # running target이 정해지고, 남은 source domain에서 class sample된 테이블에서 가장 작은 값.
    # split test & (training and val) set
    # fixed target domains will be all in testlist.
    # (training and val) set will be same number with img_num*(1-test_rate)
    numimg_val = int(img_num*val_rate)

    val_list, traintest_list = split_tetr_domclstable(cls_sampled_table,numimg_val,check_valnum_doms=fixtarget_doms)  # 모든 도메인 똑같이 validation 뽑음.
    val_df = pd.DataFrame(val_list, columns=['dom', 'cls', 'img'])  # there are no fixed target domains in train_df
    trte_df = pd.DataFrame(traintest_list, columns=['dom', 'cls', 'img'])

    target_doms = fixtarget_doms + running_target_doms
    numimg_train = int(img_num * (1 - test_rate - val_rate))
    # target doms list는 sample하지 않는다.
    train_list, test_list = split_tetr_domclstable(trte_df, numimg_train,target_doms=target_doms)

    train_df = pd.DataFrame(train_list, columns=['dom', 'cls', 'img'])  # there are no fixed target domains in train_df
    test_df = pd.DataFrame(test_list, columns=['dom', 'cls', 'img'])

    # for saving data frame to csv by changing the domain cls index to name.
    # named_test_df = replace_domcls_toname(test_df, domain_names, dom_dataset_list[0].class_to_idx)
    # named_train_df = replace_domcls_toname(train_df, domain_names, dom_dataset_list[0].class_to_idx)
    # named_val_df = replace_domcls_toname(val_df, domain_names, dom_dataset_list[0].class_to_idx)
    # named_val_df = replace_domcls_toname(val_df, domain_names, dom_dataset_list[0].class_to_idx)
    # named_train_df = replace_domcls_toname(train_df, domain_names, dom_dataset_list[0].class_to_idx)

    # print pivot table of count
    print('train dataset count table of imbalance dataset')
    print(train_df.pivot_table(index='cls', columns='dom', aggfunc='count'))
    print('val dataset count table of imbalance dataset')
    print(val_df.pivot_table(index='cls', columns='dom', aggfunc='count'))
    print('test dataset count table of imbalance dataset')
    print(test_df.pivot_table(index='cls', columns='dom', aggfunc='count'))

    return (train_df,val_df,test_df)

def split_train_val(trainval_df,val_rate, test_rate):
    # split val & train set of remain domains(not in fixed domains)
    # assume that fixed target domains are extracted from trainval_df.
    # output trainlist, vallist
    num_now = trainval_df.groupby(['dom', 'cls']).count().iloc[0, 0].item()
    num_valimgs = int(num_now*val_rate/(1-test_rate))
    vallist, trainlist = split_tetr_domclstable(trainval_df, num_valimgs)
    return (trainlist,vallist)

def replace_domcls_toname(df, domain_names, class_to_idx):
    newdf = df.copy()
    newdf['dom'] = newdf['dom'].replace(list(range(len(domain_names))), domain_names)
    newdf['cls'] = newdf['cls'].replace([v for k, v in class_to_idx.items()], class_to_idx.keys())

    return newdf

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


def get_imb_foldername_bysetting(version, dataset, cls_num, test_rate,val_rate, target_fix):
    # version is date information.(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    fixed_target_name = ''.join([str(x) for x in target_fix])
    folder_name = '_'.join([version, 'dataset', dataset, 'numcls', str(cls_num), 'testrate',
                            str(test_rate).replace('.', ''),'valrate',str(val_rate).replace('.', ''), 'fixtarget', fixed_target_name])
    return folder_name



def get_imb_csvpath_bysetting(params, runtarget,tetrvalimb,minor=0,imbrate=100,clsordom='domain'):
    # params is tuple set for common parameter.
    imb_rootdir_path,version, dataset, numcls, testrate, valrate, fixtarget = params
    # tetrvalimb is 'test' or 'train' or 'val' or 'imbtrain'
    assert (tetrvalimb in ['test', 'train','val', 'imbtrain']),tetrvalimb + 'is not in condition'

    upfolder = get_imb_foldername_bysetting(version,dataset,numcls,testrate,valrate,fixtarget)
    midfolder = get_runtarget_foldername(runtarget)

    if tetrvalimb == 'imbtrain':
        filename = get_imb_filename_bysetting(minor,imbrate,clsordom)+'.csv'
    else:
        filename = tetrvalimb+'.csv'

    return os.path.join(imb_rootdir_path,upfolder,midfolder,filename)


def get_imb_filename_bysetting(minor_domain, imb_rate, imb_type):
    # minor domain should be one digit
    return '_'.join(
        ['imbtrain','minor', str(minor_domain), 'imbrate', str(imb_rate), 'clsordom', imb_type])


def domclsimglist_to_clssampleddf(dom_dataset_list,domain_list, cls_num, fixtarget_doms):
    temp_list = []
    for domain, dataset in enumerate(dom_dataset_list):
        for img, cls in dataset.imgs:
            temp_list.append([domain, cls, img])

    # make domain class imagepath column dataframe of pandas.
    dom_cls_table = pd.DataFrame(temp_list, columns=['dom', 'cls', 'img'])

    # get maximum number of images table which correct the num_cls number.
    remain_cls_list, img_num = remaincls_numimgs_df(dom_cls_table, cls_num, fixtarget_doms)

    # sampling by remain class.
    cls_sampled_table = dom_cls_table[
        dom_cls_table['cls'].apply(lambda x: True if x in remain_cls_list else False)]

    return (replace_domcls_toname(cls_sampled_table,domain_list,dom_dataset_list[0].class_to_idx),img_num)


def get_runtarget_foldername(runtargets):
    return '_'.join(['runtarget', ''.join([str(x) for x in runtargets])])


def get_clssampled_csvpath_bysetting(params):
    # params is tuple set for common parameter.
    # return the class sampled csv path & max num img when num cls is #.
    root_imbdir, version, dataset, numcls, testrate, valrate, fixtarget = params
    upfolder = get_imb_foldername_bysetting(version, dataset, numcls, testrate, valrate, fixtarget)
    folder_path = os.path.join(root_imbdir, upfolder)
    cls_sampled_name = [x for x in os.listdir(folder_path) if re.search("ClsSampledDf_*", x)][0]
    csvpath = os.path.join(folder_path, cls_sampled_name)
    num_img = int(os.path.splitext(cls_sampled_name)[0].split('_')[-1])
    return (csvpath,num_img)

def make_trteval_csv_bysetting(params,dataroot_dir,running_targets):
    # datarootdir is domain generalization data root dir
    # running target is target domain.
    # runtarget 세팅에 따라 없으면 만든다.
    # 해당하는 버전의 데이터셋이 없으면 만든다.

    imb_rootdir_path, version, dataset, numcls, testrate, valrate, fixtarget = params
    folder_name = get_imb_foldername_bysetting(version, dataset, numcls, testrate, valrate,
                                               fixtarget)
    folder_path = os.path.join(imb_rootdir_path, folder_name)
    try:
        os.makedirs(folder_path)
        datapath = dataset_path(dataroot_dir, dataset)
        # read domain list from data path
        dom_dataset_list, domain_list = datapath_to_domaindatasetlist(datapath)
        # get class sampled domain & maximum train image number.
        cls_sampled_df, img_num = domclsimglist_to_clssampleddf(dom_dataset_list, domain_list, numcls,
                                                                fixtarget)
        cls_sampled_csvpath = os.path.join(folder_path, 'ClsSampledDf_imgnum_' + str(img_num) + '.csv')
        cls_sampled_df.to_csv(cls_sampled_csvpath, index=False)

    except FileExistsError:
        # make folder per runtargets
        cls_sampled_csvpath,_ = get_clssampled_csvpath_bysetting(params)
        print(cls_sampled_csvpath, 'exist')

    runtarget_folder_name = get_runtarget_foldername(running_targets)
    runtarget_folder_path = os.path.join(folder_path, runtarget_folder_name)

    try:
        os.makedirs(runtarget_folder_path)
        train_df, val_df, test_df = mk_imbalance_newdatacsv(cls_sampled_csvpath, testrate, valrate,
                                                            fixtarget, running_targets)
        train_df.to_csv(os.path.join(runtarget_folder_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(runtarget_folder_path, 'test.csv'), index=False)
        val_df.to_csv(os.path.join(runtarget_folder_path, 'val.csv'), index=False)

    except FileExistsError:
        print(runtarget_folder_path, 'exist')
        return runtarget_folder_path

    return runtarget_folder_path

def make_imbtrain_csv_bysetting(params, runtarget, minor, imbrate, clsordom):
    # params is tuple set for common parameter.
    imb_rootdir_path, version, dataset, numcls, testrate, valrate, fixtarget = params

    upfolder = get_imb_foldername_bysetting(version, dataset, numcls, testrate, valrate, fixtarget)

    folder_path = os.path.join(imb_rootdir_path, upfolder)

    # search the class smapled csv file from imbalance folder

    cls_sampled_name = [x for x in os.listdir(folder_path) if re.search("ClsSampledDf_*", x)][0]
    # get domain name list
    cls_sampled_csvpath = os.path.join(folder_path, cls_sampled_name)
    cls_sampled_df = pd.read_csv(cls_sampled_csvpath)
    domain_namelist = cls_sampled_df.groupby('dom').count().index.to_list()

    # get runtarget_# folder to dataframe and make imbalance.
    runtarget_foldername = get_runtarget_foldername(runtarget)
    runtarget_folderpath = os.path.join(folder_path, runtarget_foldername)
    train_csv_path = os.path.join(runtarget_folderpath, 'train.csv')
    imb_df = csv_to_imbalance_csv(train_csv_path, clsordom, imbrate, minor, domain_namelist)
    imb_df.to_csv(
        os.path.join(runtarget_folderpath, get_imb_filename_bysetting(minor, imbrate, clsordom) + '.csv'),
        index=False)
    return os.path.join(runtarget_folderpath, get_imb_filename_bysetting(minor, imbrate, clsordom) + '.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make a imbalance dataset csv')
    parser.add_argument('--command', choices=['totalnew', 'anotherimb']) # totalnew is making new random dataset csv(include test dataset)
    parser.add_argument('--imb_rate', type=int, default=5)
    parser.add_argument('--cls_num', type=int, default=5)
    parser.add_argument('--test_rate', type=float, default=0.2)
    parser.add_argument('--val_rate', type=float, default=0.1)
    parser.add_argument('--minor_domain', type=int, default=3)
    parser.add_argument('--running_targets', type=int, nargs='+', default=[1])  # 도메인 이름 알파벳으로 소팅하고 난뒤 index로 타겟 도메인 지
    parser.add_argument('--imb_data_rootdir', type=str, default="../imbalance_result_output")
    parser.add_argument('--dataroot_dir', type=str,default="/home/genie/2hddb/dg_dataset")
    parser.add_argument('--version', type=str,default="20201109235008")
    parser.add_argument('--dataset', choices=datasets.DATASETS)
    parser.add_argument('--target_fix', type=int, nargs='+', default=[])
    parser.add_argument('--imb_type', choices=['class','domain']) # 'class_or_domain' -> imb_type
    args = parser.parse_args()



    if args.command == 'totalnew':
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        params = (args.imb_data_rootdir,now, args.dataset,args.cls_num,args.test_rate,args.val_rate,args.target_fix)
        make_trteval_csv_bysetting(params,args.dataroot_dir,args.running_targets)

    elif args.command == 'anotherimb': # make imbalance data from train csv

        params = (args.imb_data_rootdir, args.version, args.dataset, args.cls_num, args.test_rate, args.val_rate,
                  args.target_fix)
        make_imbtrain_csv_bysetting(params, args.running_targets, args.minor_domain, args.imb_rate, args.imb_type)
