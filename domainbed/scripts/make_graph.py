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
    legend_elements = [Line2D([0], [0], color='r', lw=4, label=col1legend),
                       Line2D([0], [0], color='b', lw=4, label=col2legend)]
    fig, ax1 = plt.subplots(figsize=(8, 7))
    if rotate:
        plt.xticks(rotation=-45)
    sns.lineplot(x=xaxis, y=col1, data=tempdf, color='r', ax=ax1)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    sns.lineplot(x=xaxis, y=col2, data=tempdf, color='b', ax=ax2)
    ax3.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)
    if barhue:
        sns.barplot(x=xaxis, y=col3,data=tempdf, ax=ax3, alpha=0.2, hue='hue',dodge=False)
    else:
        sns.barplot(x=xaxis, y=col3,data=tempdf, ax=ax3, alpha=0.2, color='grey')
    plt.legend([],[], frameon=False)
    ax1.legend(handles=legend_elements,loc='upper right')
    ax1.grid()
    ax2.grid()
    sns.set_style('white')
    ax3.set_ylabel(col3legend)
    plt.title(title)
    plt.tight_layout()
    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0.01, 0.275)
    if barlimit!=None:
        ax3.set_ylim(barlimit[0], barlimit[1])
    if savepath != None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

def mkdictperstep(csvdf,popul):
    steps = list(set(csvdf['step'].to_list()))
    csvdf['domcls'] = csvdf['domain'] + '_' + csvdf['class']
    popul['domcls'] = popul['domain'] + '_' + popul['class']
    stepdict_domclsdf = {}
    stepdict_clsdf = {}
    stepdict_domdf = {}
    stepdict_clsperdomdf = {}

    dompopul = popul.groupby(by='domain').count()['image']
    clspopul = popul.groupby(by='class').count()['image']
    domclspopul = popul.groupby(by='domcls').count()['image']
    clsperdompopul = popul.groupby(by=['domain', 'class']).count()['image'].reset_index()

    for step in steps:
        stepdf = csvdf[csvdf['step']==step]

        domdf = stepdf.groupby(by='domain').mean()
        a = domdf.reset_index()
        a['hue'] = a['domain']
        domdf = a.set_index('domain')
        domdf = pd.concat([domdf, dompopul], axis=1).dropna().sort_values(by='image',ascending = False)
        domdf = domdf.reset_index().rename(columns={'index': 'domain'})


        clsdf = stepdf.groupby(by='class').mean()
        clsdf = pd.concat([clsdf, clspopul], axis=1).dropna().sort_values(by='image',ascending = False)
        clsdf = clsdf.reset_index()

        domclsdf = stepdf.groupby(by='domcls').mean()
        a = domclsdf.reset_index()
        a['hue'] = a['domcls'].str.slice(stop=1)
        domclsdf = a.set_index('domcls')
        domclsdf = pd.concat([domclsdf, domclspopul], axis=1).dropna().sort_values(by='image',ascending = False)
        domclsdf = domclsdf.reset_index().rename(columns={'index': 'dom_cls'})

        stepdict_domclsdf[step] = domclsdf
        stepdict_clsdf[step] = clsdf
        stepdict_domdf[step] = domdf

        stepdict_clsperdomdf[step] = {}
        for dom in list(set(stepdf['domain'].to_list())):
            stepdomdf = stepdf[stepdf['domain']==dom].groupby(by='class').mean()
            stepdomdf['hue'] = dom
            x =  clsperdompopul[clsperdompopul['domain']==dom].groupby(by='class').mean()
            stepdict_clsperdomdf[step][dom] = pd.concat([stepdomdf,x], axis=1).dropna().sort_values(by='image', ascending=False).reset_index()


    return (stepdict_domdf,stepdict_clsdf,stepdict_domclsdf,stepdict_clsperdomdf)


if __name__=='__main__':

    csvpath = '/home/genie/PycharmProjects/DomainBed/search/269ba7ac8d3fa0912b2147b2649978ad/lossinfo_per_domcls.csv'
    data = 'pacs'
    img_savepath = '/home/genie/PycharmProjects/DomainBed/search/269ba7ac8d3fa0912b2147b2649978ad/graph'
    globalsizelimit = (750,2200)
    localsizelimit = (100,500)


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

    col1, col2 = 'accuracy', 'weight'

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

    stepdict_domdf, stepdict_clsdf, stepdict_domclsdf, stepdict_clsperdomdf = mkdictperstep(df,popul)


    for step in tqdm(sorted(stepdict_domdf.keys())):
        if step !=0:
            stepstr = str(prevstep) + '~' + str(step)
        else:
            stepstr = str(step)
        prevstep = step
        title = 'Domain vs mean Acc Weight Step ' + stepstr
        savepath = os.path.join(img_savepath,title+'.png')
        overlab_graph(stepdict_domdf[step], col1, col2, 'image', title,xaxis='domain',savepath=savepath,rotate=False,barlimit = globalsizelimit)

        title = 'Class vs mean Acc Weight Step ' + stepstr
        savepath = os.path.join(img_savepath,title+'.png')
        overlab_graph(stepdict_clsdf[step], col1, col2, 'image', title,xaxis='class',savepath=savepath,barlimit = globalsizelimit)

        title = 'Domain_Class vs mean Acc Weight Step ' + stepstr
        savepath = os.path.join(img_savepath,title+'.png')
        overlab_graph(stepdict_domclsdf[step], col1, col2, 'image', title,xaxis='dom_cls',savepath=savepath,barhue=True,barlimit = localsizelimit)

        for k,v in stepdict_clsperdomdf[step].items():
            title = 'Domain:' + k+' Class vs mean Acc Weight Step ' + stepstr
            savepath = os.path.join(img_savepath,title+'.png')
            overlab_graph(v, col1, col2, 'image', title,xaxis='class',savepath=savepath,barlimit = localsizelimit)
