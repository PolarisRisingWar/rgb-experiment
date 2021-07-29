from itexperiments import experiment

import configparser

def value_type(key):
    """
    输入key，根据方括号里面的值来返回value的type
    """
    #print(key)
    left_index=key.find('[')
    assert left_index>0
    right_index=key.find(']')
    #print(right_index)
    #print(len(key))
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
    #print(key)
    #print(value_type(key))
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