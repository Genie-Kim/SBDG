import pandas as pd


# column is domain, row is class, here class will be index
df = pd.read_csv('/home/genie/1hddc/DG dataset/domainnet/domainnet.csv', index_col=0)
# df = pd.read_csv('/home/genie/domainnet_test.csv', index_col=0)



while(1):
    minncls = int(input('\n\ntype minimum # of remain classes you want(int) ==>'))
    dropped_class = []
    newdf = df
    print(newdf.head(4))
    drop_domains = input('\n\ntype domain want to drop(ex : x_x_x)').split('_')
    if drop_domains != ['']:
        newdf = newdf.drop(drop_domains,axis=1)
    while(len(newdf)>minncls):
        newdf.min().idxmin()
        minimum_class = newdf.loc[:, newdf.min().idxmin()].idxmin()
        newdf = newdf.drop(minimum_class,axis=0)
        dropped_class.append(minimum_class)
    print(newdf.min())
    print('dropped classes : '+', '.join(dropped_class))
    print('remain classes : '+(', '.join(newdf.index.tolist())))
    print('dropped classes number = ',len(dropped_class))

    minimum =newdf.min().min()
    print('minimum # of images = ' ,minimum)
    D = len(newdf.columns)
    print('# of images of balance dataset = ',minimum*(D-1.9)/(D-1))
    import pdb; pdb.set_trace()




