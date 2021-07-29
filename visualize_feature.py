import torch
from torch.functional import Tensor

from sklearn.decomposition import PCA

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from initial_params import InitialParameters

def visualize_feature(x,
                    y=None,
                    pics_root=InitialParameters.default_pics_path,
                    pic_name='pic1.png',
                    title="可视化特征图像"):
    """绘制x的PCA二维可视化图像。如果有y的话就按照对应的标签来绘制颜色"""
    font = FontProperties(fname=InitialParameters.simhei_ttf_path)
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

    if isinstance(x,Tensor):
        x=x.cpu()

    if isinstance(y,Tensor):
        y=y.cpu()
    
    pca=PCA(n_components=2)
    x=pca.fit_transform(x)
    plt.scatter(x[:,0],x[:,1],c=y)
    #TODO:cmap试了几种，在二分类标签上都不好看
    #考虑按照分类数不同来用不同的内置colormap，或者直接自定义

    #TODO:plt的title也可以自定义
    plt.title(title,fontproperties=font)

    plt.savefig(pics_root+'/'+pic_name)

    plt.close()
    
#visualize_feature示例
"""
torch_rng=torch.Generator()
torch_rng.manual_seed(2147483647)
x_example=torch.rand((100,10),generator=torch_rng)
y_example=torch.randint(0,10,(100,),generator=torch_rng)

visualize_feature(x_example,y_example)
"""