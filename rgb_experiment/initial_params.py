import os

father_dir_path=os.path.dirname(os.path.realpath(__file__))  #rgb_experiment的路径
grandfather_dir_path=os.path.dirname(father_dir_path)  #lifelong-experiment的路径

class InitialParameters:
    default_data_path=r'/data/wanghuijuan/dataset2/rd2pd_ds'  #rd2pd格式数据存放路径
    default_pics_path=grandfather_dir_path+r'/pics'  #默认图片输出路径
    simhei_ttf_path=father_dir_path+r'/SimHei.ttf'  #黑体字ttf文件存储路径
    tnr_ttf_path=father_dir_path+'/times.ttf'  #Times New Roman ttf文件存储路径

    #以下数据集名称对应的数据集都放在default_data_path下子文件夹同数据集名的位置处
    dataset_names1=['cora','citeseer','pubmed','Github','Elliptic','Film','Wiki','Clothing',
                    'Electronics','Dblp','Yelpchi','Alpha','Weibo','bgp','ssn5','ssn7',
                    'chameleon','squirrel','Aids','Nba','Wisconsin','Texas','Cornell',
                    'Pokec_z','ogbn_arxiv']
    dataset_names2=['ssn1','ssn2','ssn3','ssn5','ssn6']
    #dataset_names2是搞得不太好的图数据，一般不用

    #注意以下两个参数之间是要一一对应的，之所以没合一起主要是因为那样就太长了
    #9个模型+C&S
    model_names=['MLP','GCN','GraphSAGE','GAT','GGNN','APPNPStack','GraphSAGE2','PTA',
                'DAGNN','SuperGAT','SGC']
    default_init_params=[{'num_layers': 3, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'num_layers': 2, 'hidden_unit': 8, 'dropout_rate': 0.5, 'heads': 8},
                        {'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'hidden_unit': 64, 'dropout_rate': 0.5, 'alpha': 0.1, 'K': 10},
                        {'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'nhid':64,'dropout':0,'epsilon':100,'mode':2,'K':10,'alpha':0.1},
                        {'hidden_dim':64,'K':10,'dropout_rate':0.5},
                        {'hidden_dim':8,'heads':8,'dropout_rate':0.6,'edge_sample_ratio':0.8,'neg_sample_ratio':0.5},
                        {'K':2}]
    #APPNP的α和K都是原论文的设置，PTA, DAGNN的参数都来自原论文的GitHub项目
    #SuperGAT的超参参考自https://github.com/pyg-team/pytorch_geometric/blob/master/examples/super_gat.py
    #SGC的超参设成2是因为我调了几个数据集综合起来感觉2的表现真的不错

    default_cs_param={'num_correction_layers': 50, 'correction_alpha': 0.8, 'num_smoothing_layers': 50, 'smoothing_alpha': 0.8, 'autoscale': True}
    #参数参考自：
    #https://github.com/rusty1s/pytorch_geometric/blob/master/examples/correct_and_smooth.py