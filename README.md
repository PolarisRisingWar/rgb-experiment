本项目/文件夹只包含代码和配置文件等。数据文件未包含。

[TOC]
（注：我用VSCode的MPE边看边写的，别的Markdown预览方式可能不支持部分语法）

# 1. quick start
## 1.1 调用函数的方式
1. 本项目在后续开发过程中可能会上pypi，但是暂时还是一个单纯的package，需要手动引入目录后才能调用包：
```python
import sys
sys.path.extend(['您clone的GitHub项目的路径'])
```
2. 本项目的部分默认参数设置目前储存于 `rgb_experiment/initial_params.py` 文件中，如果您需要修改相应参数，需要进入文件进行修改。
3. 本项目的核心函数为 `experiment()` 函数，同时也提供了其他使用的辅助功能。
调入函数的代码为：
```python
from rgb_experiment import experiment
```
4. 实验中需要 `torch_geometric.data.Data` 格式的图数据，您可以通过`experiment()` 内置的功能通过 `RD2PD` 类导入，也可以通过其他方式直接得到Data数据并传入 `experiment()` 函数。对数据格式的要求等注意事项见本节下的1.2部分。
    4.1 隐式调用 `RD2PD` 在 `experiment()` 中传入数据（见 `examples/simple_example.py` 示例）
    4.2 显式传入Data数据（以使用 `torch_geometric.datasets` 模块传入数据为例，见 `examples/explictly_data_example.py` 代码示例）
5. 除直接通过代码传入参数外，也可以通过配置文件传入参数（代码见示例文件 `examples/ini_example.py` 示例，配置文件见 `ini_files` 文件夹下示例）
## 1.2 引用数据的方式
1. experiment()函数中数据的默认路径可通过initial_params.py中 `default_data_path` 参数修改。
2. 符合 `RD2PD` 类要求的图数据集原始数据：可直接将文件根目录路径传入 `experiment()`
目录下至少应装载如下数据：
> x.npy（节点特征，`np.save()` 结果，要求元素数据格式为整数，尺寸为 `[num_node, feature_dim]`）
y.npy（节点标签，`np.save()` 结果，要求元素数据格式为整数，尺寸为 `[num_node]`，元素为-1时表示无标签节点）
edge_index.npy（边，`np.save()` 结果，要求元素数据格式为float。无向图的尺寸为 `[2, 2*num_edge]`，有向图的尺寸为 `[2,num_edge]`。在代码中是通过该变量调用 `torch_geometric` 的API来判断一个图是否是有向图的）

本项目中使用 `numpy.ndarray` 格式的数据是为了给使用PyTorch、TensorFlow等各种包的用户都提供同样简洁明了的数据格式。

4. `torch_geometric.data.Data` 格式数据：可在设置入参 `specify_data=True` 后，直接将数据传入 `experiment()` 函数的 `data` 入参。
您可以提前在data中添加 `train_mask / val_mask / test_mask` 属性（要求是尺寸为 `[num_nodes]` 的布尔Tensor），也可以通过设置 `remake_data_mask=True` 来命令 `experiment()` 自动配置相应属性。
## 1.3 增加模型的方式
1. 模型文件都放在了 `rgb_experiment.models` 文件夹下，对一般仅需改模型架构的模型而言，可直接将以 `torch.nn.Module` 为基类的模型类文件放在该文件夹中，并在该文件夹下的`__init__.py` 中引用模型，然后在 `exeriment()` 中引用该模型，并在 `if model_name==` 判断语句中增加对该模型的处理即可。
2. 一般来说，您的模型需要返回一个字典，其中键x的value是输出值，键emb的value建议是输出层前一层的节点嵌入值。
3. 如果您对模型有特殊需求，可以参考PTA模型和C&S模型的情况，来定制化地完成一些工作。
4. 请注意：本实验中模型名称在 `experiment()` 调用中是不区分字母大小写的（见 `itexperiment.py`）。

# 2. package `rgb_experiment` 的简易文档
详细内容可直接参照各文件的注释。
1. `func experiment()`：传入参数，返回GNN实验指标字典。
2. `class RD2PD`：传入放置原始数据文件的参数，返回 `torch_geometric.Data` 格式的数据。
3. `func visualize_feature()`：绘制入参中 `x` 经PCA降维到2维的图像。
4. `initial_params.py`：修改默认数据设置。
5. `submodule models`：模型
已可用的模型含：（以下按模型名字母顺序排序。模型的具体结构可参考对应代码）
    1. APPNPStack（APPNP模型，原论文 [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997)）
    2. DAGNN（原论文 [Towards Deeper Graph Neural Networks](https://www.kdd.org/kdd2020/accepted-papers/view/towards-deeper-graph-neural-networks)）
    3. GAT（原论文 [Graph Attention Networks](https://arxiv.org/abs/1710.10903)）
    4. GCN（原论文 [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)）
    5. GGNN（原论文 [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493)）
    6. GraphSAGE（原论文 [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)）（应用 `torch_geometric.nn.MessagePassing` 基类实现卷积层）
    7. GraphSAGE2（同GraphSAGE）（应用 `torch_geometric.nn.SAGEConv` 类作为卷积层，在较大的数据集上有OOM的问题）
    8. MLP
    9. PTA（原论文 [On the Equivalence of Decoupled Graph Convolution Network and Label Propagation](https://arxiv.org/abs/2010.12408)）
    10. （另需注意：本项目也实现了C&S模型，但是在 `experiment()` 函数中直接进行了post-processing，就不在这一部分）
6. `submodule utils`：一些实用的函数
    1. 数据集划分，即在 `data` 中增加 `train / val / test mask`（`y=-1` 即认为是无标签节点，不参与数据集划分）
        1. `get_whole_mask()`：对分类任务的数据，按比例对所有数据进行划分（保证训练集中含有每一类标签的节点）
        2. `get_classification_mask()`：对分类任务的数据的每一类节点，按比例对所有数据进行划分
        3. TODO：对分类任务的数据的每一类节点选取特定数量的节点作为训练集、验证集及测试集（参考 `torch_geometric.datasets.Planetoid` 的随机数据划分方式）
    2. TODO：通过节点序列或边序列返回对应的子图
    3. TODO：图采样
    4. TODO：自动调参


# 2. 本项目中其他各文件的简介
1. `README.md` 介绍项目情况
2. `最终结果.csv`：如题，是各数据集上的baseline accuracy指标结果。OOM就是对应数据集在对应模型上OOM了。
（注意1：对有向图数据，额外运行了将该数据转换为无向图后得到的结果）
（注意2：不实时更新，需要最新版的结果建议直接联系作者）
3. 文件夹 `examples`：示例代码
    1. `simple_example.py`：直接调用 `experiment()` 函数得到结果并打印。
    2. `explictly_data_example.py`：显式传入 `torch_geometric.data.Data` 格式的图数据（通过 `torch_geometric.datasets` 模块导入）到 `experiment()` 函数中，得到结果并打印。
    3. `ini_example.py`：通过配置文件传参。
    4. `rd2pd_example.py`：通过 `RD2PD` 类导入 `torch_geometric.data.Data` 格式的图数据。
    5. `auto_tune_hp.py`：一个自动调参的函数（但是不太好用）。使用方法可参考 `try_auto_tune_hp.py`
    6. `try_auto_tune_hp.py`：见3.5介绍
    7. 文件夹 `outdated_entirely_run_example`：在前几个版本的本项目结构中使用的跑所有数据集在所有模型上的指标结果的代码及相应的输出内容，由于代码结构发生变化已无法使用，但可作为参考
        1.  `run_example1`：直接用 `initial_params.py` 中内置的数据集名称和路径，调用 `experiment()` 函数运行结果
        2. `run_example2`：用外部传入的 `Data` 文件，调用 `experiment()` 函数运行结果
4. 文件夹 `ini_files`：配置文件示例
5. 文件夹 `pics`：默认图片输出路径，预先内置输出图片示例

# 3. 其他注意事项
1. 本项目中matplotlib绘图默认使用本项目下的黑体文件（SimHei.ttf），在代码里写死了。
2. 各模型必须传入 `experiment()` 的参数 `model_init_param` 键及格式见各模型文件，也可以直接参考 `initial_params.py` 里的示例。

# 4. 致谢及其他
1. 感谢同组小伙伴的技术支持！
2. 本项目的requirements将在随后给出。
3. 部分模型的参数等配置参考了原始论文配套代码或torch_geometric项目提供的例子及集成函数。