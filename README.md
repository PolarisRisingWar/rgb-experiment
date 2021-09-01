本项目/文件夹只包含代码和配置文件等。数据文件未包含。

[TOC]
（注：我用VSCode的MPE边看边写的，别的Markdown预览方式可能不支持部分语法）

# 1. quick start
## 1.1 调用函数的方式
1. 本项目在后续开发过程中可能会上pypi，但是暂时还是一个单纯的package，需要手动引入目录后才能调用包：
```python
import sys
sys.path.extend(['您clone的GitHub项目的路径'])
import rgb_experiment
```
2. 本项目的部分默认参数设置目前储存于 `rgb_experiment/initial_params.py` 文件中，如果您需要修改相应参数，需要进入文件进行修改。本项目计划后期修改为通过工作路径下的配置文件对这些数据进行传参，但是目前还需要在py文件内修改。
3. 本项目的核心函数为 `experiment()` 函数，同时也提供了其他使用的辅助功能。
调入函数的代码为：
```python
from rgb_experiment import experiment
```
4. 实验中需要 `torch_geometric.data.Data` 格式的图数据，您可以通过`experiment` 内置的功能通过 `zjutoid` 类导入，也可以通过其他方式直接得到Data数据并传入 `experiment` 函数。对数据格式的要求等注意事项见本节下的1.2部分。<br>
    4.1 隐式调用 `zjutoid` 在 `experiment` 中传入数据（见 `examples/implicitly_zjutoid_example.py` 示例）
    4.2 显示传入Data数据（见 `examples/explictly_data_example.py` 示例）
5. 除直接通过代码传入参数外，也可以通过配置文件传入参数（代码见示例文件 `examples/ini_example.py` 示例，配置文件见 `ini_files` 文件夹下示例）
## 1.2 引用数据的方式
1. experiment()函数中数据的默认路径可通过initial_datapath.py中 `default_data_path` 参数修改。
2. 类Planetoid格式，可使用zjutoid.py和iozjutoid.py处理：可直接将文件根目录路径传入 `experiment()`
要求目录root下raw文件夹下装载数据文件，root文件夹名即为数据集名称，数据文件中间名需要是数据集名称的全小写格式。
以Github数据下raw文件夹中存放的原始数据为例：
> ind.github.ally
ind.github.test.index
ind.github.ty
ind.github.y
ind.github.allx
ind.github.graph
ind.github.tx
ind.github.x

通过zjutoid类调用数据的代码见 `examples/zjutoid_example.py` 示例
3. `x.pt` &nbsp; `y.pt` &nbsp; `edge_index.pt`（Tensor.save()结果）形式数据可使用rd2pd.py 中的类 RD2PD 处理为 `torch_geometric.data.Data` 格式。详情可参考该文件注释。
4. `torch_geometric.data.Data` 格式数据：可直接将数据传入 `experiment()`。
您可以提前在data中添加 `train_mask/val_mask/test_mask` 属性（要求是尺寸为 [num_nodes] 的布尔tensor），也可以通过设置 `remake_data_mask=True` 来命令 `experiment()` 自动配置相应属性。
## 1.3 增加模型的方式
1. 模型文件都放在了 `.models` 文件夹下，可直接将含有以 `torch.nn.Module` 为基类 的模型类文件放在该文件夹中，并在该文件夹下的`__init__.py` 中引用模型，然后在 `exeriment()` 中引用该模型，并在 `if model_name==` 判断语句中增加对该模型的处理即可。
2. 一般来说，您的模型需要返回一个字典，其中键x的value是输出值，键emb的value建议是输出层前一层的节点嵌入值。
3. 如果您对模型有特殊需求，可以参考PTA模型的情况，来定制化地完成一些工作

# 2. 本项目中各文件的简介
1. README.md 介绍项目情况
2. zjutoid.py 和 iozjutoid.py 源自奕青同学（我对其中划分数据集部分代码做了修改），可用于处理类Planetoid格式的图数据集文件。
3. zjutoid2.py 是我写的，见本文档1.1部分介绍。
3. initial_params.py 用于初始化一些数据集的路径（建议使用以防造成更改数据集位置而需要大量修改代码的问题）
4. 文件夹 ini_files：放置示例配置文件，可使用配置文件向 `experiment()` 函数传入参数。使用配置文件的示例见 ini_example.py
5. 文件夹 pics：默认输出图片放置位置
6. itexperiments.py：内含 `experiment()` 函数及其使用的 `test()` 和 `compare_pred_label()` 函数
7. visualize_feature.py：内含 `experiment()` 函数中调用的 `visualize_feature()` 函数，用于绘制节点特征降维到2维平面上的图像。
8. auto_tune_hp.py：内含可调用的自动调参函数（其实不好用），使用示例见 try_auto_tune_hp.py
9. run_example1.py：一个典型的对 initial_params.py 中加载的所有数据集在所有模型上运行 `experiment()` 得到baseline结果的示例代码文件（输出历史见run_example1_output.out）
11. run_example2.py：类似
10. 最终结果.csv：如题，是各数据集上的baseline accuracy指标结果。OOM就是对应数据集在对应模型上OOM了。（注意1：对有向图数据，额外运行了将该数据转换为无向图后得到的结果）（注意2：不实时更新，需要最新版的结果建议直接联系作者）

# 3. 其他注意事项
1. 本项目中matplotlib绘图默认使用本项目下的黑体文件（SimHei.ttf），在代码里写死了。
2. 各模型必须传入 `experiment()` 的参数 `model_init_param` 键及格式见各模型文件。

# 4. 致谢及其他
1. 感谢黄学长和同组的小伙伴提供陪伴和支持！
2. 本项目的requirements将在随后给出。
3. 部分模型的参数等配置参考了原始论文配套代码或torch_geometric项目提供的例子及集成函数。