import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os
from tqdm import tqdm

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def overlab_graph(tempdf,col1,col2,col3,title,barlimit=None,xaxis = 'index',col1legend='Accuracy',col2legend='Weight',col3legend='# of Images',savepath=None,rotate = True,barhue = False):

    legend_elements = [Line2D([0], [0], color='r',marker = 'o', lw=1, label=col1legend,mew=0.01),
                       Line2D([0], [0], color='b',marker = 'd', lw=1, label=col2legend,mew=0.01)]
    palette = {"V": "C0", "L": "C1", "C": "C2", "S": "C3"}# domain color map
    fig, ax1 = plt.subplots(2,1,sharex=True,figsize=(5, 4), dpi=300,gridspec_kw={'height_ratios': [3, 1]})
    # fig, ax1 = plt.subplots(2,1,sharex=True,figsize=(4, 3), dpi=300,gridspec_kw={'height_ratios': [3, 1]})
    # fig.suptitle(title)
    # ax1[0].set_title('Accuracy vs Weight')
    # ax1[1].set_title('Histogram')

    if rotate:
        plt.xticks(rotation=-45)
    sns.lineplot(x=xaxis, y=col1, data=tempdf, color='r', ax=ax1[0],marker="o")
    ax2 = ax1[0].twinx()
    sns.lineplot(x=xaxis, y=col2, data=tempdf, color='b', ax=ax2,marker="d")
    # ax1[1].spines["right"].set_position(("axes", 1.2))
    # make_patch_spines_invisible(ax1[1])
    # ax1[1].spines["right"].set_visible(True)
    if barhue:
        sns.barplot(x=xaxis, y=col3,data=tempdf, ax=ax1[1], hue='hue',dodge=False,palette=palette)
        ax1[1].legend().set_title('')
        ax1[1].get_legend().remove()
    else:
        sns.color_palette("tab10")
        sns.barplot(x=xaxis, y=col3,data=tempdf, ax=ax1[1],color='tab:blue')
    # plt.legend([],[], frameon=False)
    # ax1[0].legend(handles=legend_elements,loc='upper right',fontsize='xx-small')
    ax1[1].legend(fontsize='xx-small')
    # ax1[0].grid()
    # ax2.grid()
    sns.set_style('white')
    ax1[1].set_ylabel(col3legend)
    plt.tight_layout()
    # ax1.set_ylim(0, 1.2)
    # ax2.set_ylim(0.01, 0.275)
    if barlimit!=None:
        ax1[1].set_ylim(barlimit[0], barlimit[1])
    if savepath != None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

def mkdictperstep(csvdf,popul):
    steps = list(set(csvdf['Step'].to_list()))
    csvdf['Domcls'] = csvdf['Class']+'('+csvdf['Domain'] + ')'
    popul['Domcls'] = popul['Class']+'('+popul['Domain'] + ')'
    stepdict_domclsdf = {}
    stepdict_clsdf = {}
    stepdict_domdf = {}
    stepdict_clsperdomdf = {}

    dompopul = popul.groupby(by='Domain').count()['Image']
    clspopul = popul.groupby(by='Class').count()['Image']
    domclspopul = popul.groupby(by='Domcls').count()['Image']
    clsperdompopul = popul.groupby(by=['Domain', 'Class']).count()['Image'].reset_index()

    for step in steps:
        stepdf = csvdf[csvdf['Step']==step]

        domdf = stepdf.groupby(by='Domain').mean()
        a = domdf.reset_index()
        a['hue'] = a['Domain']
        domdf = a.set_index('Domain')
        domdf = pd.concat([domdf, dompopul], axis=1).dropna().sort_values(by='Image',ascending = False)
        domdf = domdf.reset_index().rename(columns={'index': 'Domain'})


        clsdf = stepdf.groupby(by='Class').mean()
        clsdf = pd.concat([clsdf, clspopul], axis=1).dropna().sort_values(by='Class',ascending = False)
        clsdf = clsdf.reset_index()

        domclsdf = stepdf.groupby(by=['Domcls','Domain']).mean()
        a = domclsdf.reset_index()
        a['hue'] = a['Domain']
        domclsdf = a.set_index('Domcls')
        domclsdf = pd.concat([domclsdf, domclspopul], axis=1).dropna().sort_values(by='Image',ascending = False)
        domclsdf = domclsdf.reset_index().rename(columns={'index': 'Dom_Cls'})

        stepdict_domclsdf[step] = domclsdf
        stepdict_clsdf[step] = clsdf
        stepdict_domdf[step] = domdf

        stepdict_clsperdomdf[step] = {}
        for dom in list(set(stepdf['Domain'].to_list())):
            stepdomdf = stepdf[stepdf['Domain']==dom].groupby(by='Class').mean()
            stepdomdf['hue'] = dom
            x =  clsperdompopul[clsperdompopul['Domain']==dom].groupby(by='Class').mean()
            stepdict_clsperdomdf[step][dom] = pd.concat([stepdomdf,x], axis=1).dropna().sort_values(by='Class', ascending=False).reset_index()


    return (stepdict_domdf,stepdict_clsdf,stepdict_domclsdf,stepdict_clsperdomdf)


if __name__=='__main__':

    csvpath ='/home/genie/PycharmProjects/DomainBed/performance/vlcs_CMWN/vlcs3/ce8b851a7b596f05d8fa91c4bf3b55e3/lossinfo_per_domcls.csv'
    data = 'vlcs'
    img_savepath = '/home/genie/PycharmProjects/DomainBed/performance/vlcs2'
    globalsizelimit = (750,2200)
    localsizelimit = (100,500)
    extension = '.png'

    try:
        os.mkdir(img_savepath)
    except FileExistsError:
        print('there is already exist directory : ',img_savepath)
        img_savepath = img_savepath + '_new'
        os.mkdir(img_savepath)

    pacscsvpath = '/home/genie/PycharmProjects/DomainBed/pacs.csv'
    vlcscsvpath = '/home/genie/PycharmProjects/DomainBed/vlcs.csv'

    pacsnamedict = {'art_painting':'A', 'cartoon':'C','photo':'P','sketch':'S'}
    vlcsnamedict = {'Caltech101':'C', 'LabelMe':'L','SUN09':'S','VOC2007':'V'}

    col1, col2 = 'Accuracy', 'Weight'

    if data == 'pacs':
        popul = pd.read_csv(pacscsvpath)
        for k,v in pacsnamedict.items():
            popul.loc[popul['domain']==k,'domain']=v
    elif data == 'vlcs':
        popul = pd.read_csv(vlcscsvpath)
        for k,v in vlcsnamedict.items():
            popul.loc[popul['domain']==k,'domain']=v


    df = pd.read_csv(csvpath)
    df = df.drop(columns = 'Unnamed: 0')
    df.columns = map(str.capitalize, df.columns)
    popul.columns = map(str.capitalize, popul.columns)
    df['Class'] = df['Class'].str.capitalize()
    df['Domain'] = df['Domain'].str.capitalize()
    popul['Class'] = popul['Class'].str.capitalize()
    popul['Domain'] = popul['Domain'].str.capitalize()


    stepdict_domdf, stepdict_clsdf, stepdict_domclsdf, stepdict_clsperdomdf = mkdictperstep(df,popul)


    for step in tqdm(sorted(stepdict_domdf.keys())):
        if step !=0:
            stepstr = str(prevstep) + '~' + str(step)
        else:
            stepstr = str(step)
        prevstep = step
        title = 'Domain vs mean Acc Weight Step ' + stepstr
        savepath = os.path.join(img_savepath,title+extension)
        overlab_graph(stepdict_domdf[step], col1, col2, 'Image', title,xaxis='Domain',savepath=savepath,rotate=False,barhue=True)

        title = 'Class vs mean Acc Weight Step ' + stepstr
        savepath = os.path.join(img_savepath,title+extension)
        overlab_graph(stepdict_clsdf[step], col1, col2, 'Image', title,xaxis='Class',rotate=False,savepath=savepath)

        title = 'Domain_Class vs mean Acc Weight Step ' + stepstr
        savepath = os.path.join(img_savepath,title+extension)
        # drop the S_bird & change name
        tdf = stepdict_domclsdf[step]
        stepdict_domclsdf[step] = tdf.drop((tdf[tdf['Dom_Cls'] == 'Bird(S)']).index)
        overlab_graph(stepdict_domclsdf[step], col1, col2, 'Image', title, xaxis='Dom_Cls', savepath=savepath,
                      barhue=True)
        # # donot drop
        # overlab_graph(stepdict_domclsdf[step], col1, col2, 'Image', title,xaxis='Dom_Cls',savepath=savepath,barhue=True)

        for k,v in stepdict_clsperdomdf[step].items():
            title = 'Domain:' + k+' Class vs mean Acc Weight Step ' + stepstr
            savepath = os.path.join(img_savepath,title+extension)
            overlab_graph(v, col1, col2, 'Image', title,xaxis='Class',rotate=False,savepath=savepath,barhue=True)
