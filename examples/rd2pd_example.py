import sys
sys.path.extend(['whj_code2/integration_experiment'])

from rgb_experiment import RD2PD

z=RD2PD('ssn7','/data/wanghuijuan/dataset2/rd2pd_ds',specify_non_label_mask=False,
        apply_sample=False,remove_non_label_node=False,split_seed=14000094)
print(z)
print(z.data)