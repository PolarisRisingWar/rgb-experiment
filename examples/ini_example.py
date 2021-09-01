#使用配置文件进行传参的示例代码

import sys

sys.path.extend(['whj_code2/integration_experiment'])

from rgb_experiment import experiment

import configparser

def value_type(key):
    """
    输入key，根据方括号里面的值来返回value的type
    """
    left_index=key.find('[')
    assert left_index>0
    right_index=key.find(']')
    assert right_index==len(key)-1
    vt=key[left_index+1:-1]

    if vt=='.f':
        return float
    elif vt=='.i':
        return int
    elif vt=='.s':
        return str
    elif vt=='.b':
        return bool


#配置文件形式的传参示例
#TODO:支持两个以上字母的表示type的做法
config=configparser.ConfigParser()
config.read('whj_code2/integration_experiment/ini_files/ini1.ini',
            encoding='utf-8')

model_init_param={}
for key in config.options('model_init_param'):
    str_value=config.get('model_init_param',key)  #str格式的value
    model_init_param[key[:-4]]=value_type(key)(str_value)

single_params={}
for key in config.options('single_param'):
    str_value=config.get('single_param',key)  #str格式的value
    
    single_params[key[:-4]]=value_type(key)(str_value)

cs_param={}
if config.has_section('cs_param'):
    for key in config.options('cs_param'):
        str_value=config.get('cs_param',key)  #str格式的value
        cs_param[key[:-4]]=value_type(key)(str_value)

acc_dict=experiment(model_init_param=model_init_param,cs_param=cs_param,**single_params)
print(acc_dict)
#输出示例：{'ACC': 0.8535809018567639, 'precision_score': 0.8181585437925151, 'recall_score': 0.7920084703684473, 'f1_score': 0.8034470994789795}