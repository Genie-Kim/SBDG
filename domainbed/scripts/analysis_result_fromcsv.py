import pandas as pd
from domainbed import datasets, imbalance_dataset
from collections import defaultdict
from collections import Counter
import os
def list_to_pattern(lst):
    pt_list = ['up','same' , 'down']
    pattern = []
    temp = lst[0]
    for item in lst[1:]:
        if temp < item:
            pattern.append(pt_list[0])
        elif temp > item :
            pattern.append(pt_list[2])
        else:
            pattern.append(pt_list[1])
        temp = item
    return '_'.join(pattern)

def find_imp_minor_dom(df,dom, conf=0.5 ):
    minor_list = df.groupby('minor').count().index.to_list()
    pattern_dict = defaultdict(list)
    target_imp_pattern = []
    for minor in minor_list:
        target_total_dom_minor_df = df[df['minor'] == minor]
        runtarget_list = target_total_dom_minor_df.groupby('runtarget').count().index.to_list()
        for runtarget in runtarget_list:
            target_total_dom_minor_runtarget_df = target_total_dom_minor_df[
                target_total_dom_minor_df['runtarget'] == runtarget].copy()
            if len(target_total_dom_minor_runtarget_df) < len(imbrate):
                imbrate1df = df[
                    (df['runtarget'] == runtarget) & (df['imbrate'] == 1)]
                target_total_dom_minor_runtarget_df = pd.concat([target_total_dom_minor_runtarget_df, imbrate1df],
                                                                axis=0)
            acc_list = target_total_dom_minor_runtarget_df.sort_values(by='imbrate')['acc'].to_list()
            pattern = list_to_pattern(acc_list)
            pattern_dict[(minor, dom)].append(pattern)
    for k, v in pattern_dict.items():
        if max(Counter(v).values()) / len(v) >=conf:
            print(len(v))
            for x,y in Counter(v).items():
                if y == max(Counter(v).values()):
                    target_imp_pattern.append((k,x))
    return target_imp_pattern



dataset = 'ImbalanceDomainNet'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
domain_list = list.copy(vars(imbalance_dataset)[dataset].ENVIRONMENTS)

# csv_path = '/home/genie/PycharmProjects/DomainBed/domainbed/imbalance_result_output/MLDG_dataset_DomainNet_numcls_5_testrate_02_valrate_01_fixtarget_0/MLDG_test_result_summary.csv'
# csv_path = '/home/genie/PycharmProjects/DomainBed/domainbed/imbalance_result_output/MLDG_dataset_DomainNet_numcls_5_testrate_02_valrate_01_fixtarget_0/MLDG_minor_mean_result.csv'


# csv_path = '/home/genie/PycharmProjects/DomainBed/domainbed/imbalance_result_output/MMD_dataset_DomainNet_numcls_5_testrate_02_valrate_01_fixtarget_0/MMD_test_result_summary.csv'
csv_path = '/home/genie/PycharmProjects/DomainBed/domainbed/imbalance_result_output/MMD_dataset_DomainNet_numcls_5_testrate_02_valrate_01_fixtarget_0/MMD_minor_mean_result.csv'
imbrate=[1,2,4,8,16]

df = pd.read_csv(csv_path)


total_df = df[df['cls']=='@total']
mean_df = df[df['cls']=='@mean']
total_source_imp_pattern = []
total_target_imp_pattern = []
mean_imp_pattern= []
down4pattern = []

for dom in domain_list:
    total_dom_df = total_df[total_df['dom'] == dom]
    source_total_dom_df = total_dom_df[total_dom_df['source'].apply(lambda x : True if dom in x else False)]
    if len(source_total_dom_df)>0:
        total_source_imp_pattern += find_imp_minor_dom(source_total_dom_df,dom, 0.5)

    target_total_dom_df = total_dom_df[total_dom_df['source'].apply(lambda x : False if dom in x else True)]
    if len(target_total_dom_df) > 0:
        total_target_imp_pattern += find_imp_minor_dom(target_total_dom_df, dom, 0.5)

for dom in ['@source','@target']:
    mean_dom_df = mean_df[mean_df['dom'] == dom]
    if len(mean_dom_df)>0:
        mean_imp_pattern += find_imp_minor_dom(mean_dom_df,dom, 0.5)
temp = [x for x in range(len(imbrate))]
temp.reverse()
downpattern = list_to_pattern(temp)
for runtarget,minor,dom,cls in df.groupby(by=['runtarget','minor','dom','cls']).count().index.to_list():
    temp_df = df[(df['runtarget']==runtarget) & (df['minor']==minor) & (df['dom']==dom) & (df['cls']==cls)].copy()
    if len(temp_df) < len(imbrate):
        imbrate1df = df[
            (df['runtarget'] == runtarget) & (df['imbrate'] == 1)]
        temp_df = pd.concat([temp_df, imbrate1df],axis=0)
    acc_list = temp_df.sort_values(by='imbrate')['acc'].to_list()
    pattern = list_to_pattern(acc_list)
    if pattern == downpattern:
        down4pattern.append((runtarget,minor,dom,cls))

print('source important pattern : \n',total_source_imp_pattern)
print('target important pattern : \n',total_target_imp_pattern)
print('mean important pattern : \n' ,mean_imp_pattern)
print('down 4 pattern : \n',down4pattern)
columns = ['alg', 'source', 'fixtarget', 'runtarget', 'minor', 'imbrate', 'dom', 'trteval', 'cls', 'acc']
runtarget_list = df.groupby(by='runtarget').count().index.to_list()
newdf = pd.DataFrame(columns=columns)
for runtarget in runtarget_list:
    rundf = df[df['runtarget']==runtarget]
    rundf = rundf[(rundf['cls']=='@total') | (rundf['cls']=='@mean')]
    for dom,cls in rundf.groupby(by=['dom', 'cls']).count().index.to_list():
        run_dom_cls_df = rundf[(rundf['dom']==dom) & (rundf['cls']==cls)]
        imbratelist = run_dom_cls_df.groupby(by='imbrate').count().index.to_list()
        same_list = run_dom_cls_df.iloc[0,:4].to_list()
        for imbrate in imbratelist:
            meanacc = float(run_dom_cls_df[run_dom_cls_df['imbrate']==imbrate].mean(axis=0)['acc'])
            temp_df = pd.DataFrame([same_list + ['@mean', imbrate, dom,run_dom_cls_df.iloc[0,7],cls, meanacc]],
                                   columns=columns)
            newdf = pd.concat([newdf,temp_df],axis=0)

folderpath = os.path.dirname(csv_path)
# newdf.to_csv(os.path.join(folderpath,'minor_mean_result.csv'),index=False)

