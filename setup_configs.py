import yaml

from copy import deepcopy
import os
import itertools

data_dir = 'PATH_TO_DATA_DIRECTORY'
exp_dir = 'configs/experiment'
exp_fs_dir = 'configs_fs/experiment'
exp_dir_class = 'configs_class/experiment'

with open(exp_fs_dir + '/fs_template.yaml') as file:
	fs_yaml = yaml.full_load(file)

# Experiment 1 (single training env)
with open(exp_dir+'/exp1_template.yaml') as file:
    exp1_yaml = yaml.full_load(file)

train_env = ['DET_A','DET_B','DET_AB']
test_env = ['DET_A','DET_B','DET_AB']
model = ['resnet50','vit']

for xs in itertools.product(model,train_env,test_env):

	exp1_yaml_new = deepcopy(exp1_yaml)
	exp1_yaml_new['data_dir'] = data_dir
	exp1_yaml_new['model']['model']=xs[0]
	exp1_yaml_new['datamodule']['train_env']=xs[1]
	exp1_yaml_new['datamodule']['val_env']=xs[1]
	exp1_yaml_new['datamodule']['test_env']=xs[2]

	xs = list(xs)
	xs[1] = xs[1].split('_')[1]
	xs[2] = xs[2].split('_')[1]

	name = '_'.join(xs)
	name = 'exp1_'+name

	exp1_yaml_new['name'] = name
	yaml_path = os.path.join(exp_dir,name)

	with open(yaml_path+'.yaml', 'w') as outfile:
		outfile.write('# @package _global_ \n')
		outfile.write('\n')
		yaml.dump(exp1_yaml_new, outfile, default_flow_style=False,sort_keys=False)

#Experiment 2.1 (multiple training envs and no shift)
with open(exp_dir+'/exp2_template.yaml') as file:
	exp2_yaml = yaml.full_load(file)

train_env_comb = [['DET_A','DET_B'],['DET_A','DET_AB'],['DET_B','DET_AB']]
test_env = ['DET_A','DET_B','DET_AB']
model = ['resnet50','vit']

for xs in itertools.product(model,train_env_comb,test_env):

	exp2_yaml_new = deepcopy(exp2_yaml)
	xs_fs = deepcopy(xs)
	exp2_yaml_new['data_dir'] = data_dir
	exp2_yaml_new['model']['model']=xs[0]
	exp2_yaml_new['datamodule']['train_env_1']=xs[1][0]
	exp2_yaml_new['datamodule']['train_env_2']=xs[1][1]
	exp2_yaml_new['datamodule']['test_env']=xs[2]

	xs = list(xs)
	xs.append(xs[2])
	xs[1],xs[2] = (xs[1][0].split('_')[1],xs[1][1].split('_')[1])
	xs[3] = xs[3].split('_')[1]

	name = '_'.join(xs)
	name = 'exp2_'+name

	exp2_yaml_new['name'] = name
	yaml_path = os.path.join(exp_dir,name)
	with open(yaml_path+'.yaml', 'w') as outfile:
		outfile.write('# @package _global_ \n')
		outfile.write('\n')
		yaml.dump(exp2_yaml_new, outfile, default_flow_style=False,sort_keys=False)

	fs_yaml_new = deepcopy(fs_yaml)
	fs_yaml_new['datamodule_class']['train_env_1']=xs_fs[1][0]
	fs_yaml_new['datamodule_class']['train_env_2']=xs_fs[1][1]
	fs_yaml_new['datamodule_class']['test_env']=xs_fs[-1]
    
	fs_yaml_new['datamodule_pairs']['dataset_train']['env1']=xs_fs[1][0]
	fs_yaml_new['datamodule_pairs']['dataset_train']['env2']=xs_fs[1][1]
    
	fs_yaml_new['datamodule_pairs']['dataset_val']['env1']=xs_fs[1][0]
	fs_yaml_new['datamodule_pairs']['dataset_val']['env2']=xs_fs[1][1]

	yaml_path = os.path.join(exp_fs_dir,name)
	with open(yaml_path+'.yaml', 'w') as outfile:
		outfile.write('# @package _global_ \n')
		outfile.write('\n')
		yaml.dump(fs_yaml_new, outfile, default_flow_style=False,sort_keys=False)


# Experiment 2.2 and Experiment 2.3 (multiple envs with covariate shift)
shifts = [[1,5],[2,4]]
for xs in itertools.product(shifts,model,train_env_comb,test_env):

	exp2_yaml_new = deepcopy(exp2_yaml)
	xs_fs = deepcopy(xs)
	exp2_yaml_new['data_dir'] = data_dir
	exp2_yaml_new['datamodule']['shiftvar_1']=xs[0][0]
	exp2_yaml_new['datamodule']['shiftvar_2']=xs[0][1]
	exp2_yaml_new['model']['model']=xs[1]
	exp2_yaml_new['datamodule']['train_env_1']=xs[2][0]
	exp2_yaml_new['datamodule']['train_env_2']=xs[2][1]
	exp2_yaml_new['datamodule']['test_env']=xs[3]



	xs = list(xs)
	xs.append(xs[3])


	xs[2],xs[3] = (xs[2][0].split('_')[1],xs[2][1].split('_')[1])
	xs[4] = xs[4].split('_')[1]

	shift ='shift'+str(xs[0][0])+str(xs[0][1])
	xs.pop(0)
	name = '_'.join(xs)
	name = 'exp2_'+shift+'_'+name

	exp2_yaml_new['name'] = name
	yaml_path = os.path.join(exp_dir,name)
	with open(yaml_path+'.yaml', 'w') as outfile:
		outfile.write('# @package _global_ \n')
		outfile.write('\n')
		yaml.dump(exp2_yaml_new, outfile, default_flow_style=False,sort_keys=False)

	
	fs_yaml_new = deepcopy(fs_yaml)
	fs_yaml_new['datamodule_class']['train_env_1']=xs_fs[2][0]
	fs_yaml_new['datamodule_class']['train_env_2']=xs_fs[2][1]
	fs_yaml_new['datamodule_class']['test_env']=xs_fs[-1]
    
	fs_yaml_new['datamodule_pairs']['dataset_train']['env1']=xs_fs[2][0]
	fs_yaml_new['datamodule_pairs']['dataset_train']['env2']=xs_fs[2][1]
    
	fs_yaml_new['datamodule_pairs']['dataset_val']['env1']=xs_fs[2][0]
	fs_yaml_new['datamodule_pairs']['dataset_val']['env2']=xs_fs[2][1]

	yaml_path = os.path.join(exp_fs_dir,name)
	with open(yaml_path+'.yaml', 'w') as outfile:
		outfile.write('# @package _global_ \n')
		outfile.write('\n')
		yaml.dump(fs_yaml_new, outfile, default_flow_style=False,sort_keys=False)

# change dir in classifier config 
with open(exp_dir_class+'/classifier.yaml') as file:
    class_yaml = yaml.full_load(file)

class_yaml['data_dir'] = data_dir
yaml_path = exp_dir_class+'/classifier.yaml'
with open(yaml_path, 'w') as outfile:
		outfile.write('# @package _global_ \n')
		outfile.write('\n')
		yaml.dump(class_yaml, outfile, default_flow_style=False,sort_keys=False)




