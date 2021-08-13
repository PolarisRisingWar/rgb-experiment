import os

father_dir_path=os.path.dirname(os.path.realpath(__file__))

class InitialParameters:
    default_data_path=r'/data/wanghuijuan/dataset1/zjutoid_ds'
    default_pics_path=father_dir_path+r'/pics'
    simhei_ttf_path=father_dir_path+r'/SimHei.ttf'
    dataset_name_root_map=[{'dataset_name':'Github','dataset_root':default_data_path},
                            {'dataset_name':'cora','dataset_root':default_data_path},
                            {'dataset_name':'Elliptic','dataset_root':default_data_path},
                            {'dataset_name':'Film','dataset_root':default_data_path},
                            {'dataset_name':'CiteSeer','dataset_root':default_data_path},
                            {'dataset_name':'PubMed','dataset_root':default_data_path},
                            {'dataset_name':'Wiki','dataset_root':default_data_path},
                            {'dataset_name':'Clothing','dataset_root':default_data_path},
                            {'dataset_name':'Eletronics','dataset_root':default_data_path},
                            {'dataset_name':'Dblp','dataset_root':default_data_path},
                            {'dataset_name':'Yelpchi','dataset_root':default_data_path},
                            {'dataset_name':'Alpha','dataset_root':default_data_path},
                            {'dataset_name':'Weibo','dataset_root':default_data_path}]

    #注意以下两个参数之间是要一一对应的，之所以没合一起主要是因为那样就太长了
    model_names=['MLP','GCN','GraphSAGE','GAT','GGNN','APPNPStack','GraphSAGE2','PTA']
    default_init_params=[{'num_layers': 3, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'num_layers': 2, 'hidden_unit': 8, 'dropout_rate': 0.5, 'heads': 8},
                        {'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'hidden_unit': 64, 'dropout_rate': 0.5, 'alpha': 0.1, 'K': 10},
                        {'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5},
                        {'nhid':64,'dropout':0,'epsilon':100,'mode':2,'K':10,'alpha':0.1}]
    #APPNP的α和K都是原论文的设置，PTA的参数都来自原论文的GitHub项目

    default_cs_param={'num_correction_layers': 50, 'correction_alpha': 0.8, 'num_smoothing_layers': 50, 'smoothing_alpha': 0.8, 'autoscale': True}
    #参数参考自：
    #https://github.com/rusty1s/pytorch_geometric/blob/master/examples/correct_and_smooth.py