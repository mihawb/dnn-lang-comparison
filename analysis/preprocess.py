import pandas as pd
import time


def preprocess_matlab(filepath: str) -> pd.DataFrame:
	mname = filepath[filepath.rindex('_')+1:filepath.rindex('.')]
	matlab = pd.read_csv(filepath, usecols=('Epoch', 'TimeSinceStart', 'TrainingLoss', 'ValidationAccuracy', 'State'))
	matlab = matlab.rename(columns=dict(zip(matlab.columns, ['epoch','elapsed_time','loss','performance','type'])))

	matlab.insert(0, 'type', matlab.pop('type'))
	matlab.insert(4, 'elapsed_time', matlab.pop('elapsed_time'))

	matlab.loss = matlab.loss.fillna(-1)
	matlab = matlab.dropna(axis=0)
	matlab.insert(0, 'model_name', mname)

	# matlab.elapsed_time *= 1e3
	matlab = matlab[matlab.type != 'done'].drop([1])
	matlab = matlab.reset_index().drop('index', axis=1)
	matlab.type = matlab.type.replace('iteration', 'training')
	matlab.loc[1:7, 'elapsed_time'] = matlab.elapsed_time[1:-1] - matlab.elapsed_time.shift(1)[1:-1]

	return matlab


def preprocess_cudnn(results_root: str) -> pd.DataFrame:
	fcnet = pd.read_csv(f'{results_root}/cudnn_fcnet.csv')
	fcnet.model_name = 'FullyConnectedNet'

	scvnet = pd.read_csv(f'{results_root}/cudnn_scvnet.csv')
	scvnet.model_name = 'SimpleConvNet'

	return pd.concat([fcnet, scvnet])


def get_results(results_root: str, save: bool=False, full: bool=True, **frameworks) -> tuple[pd.DataFrame, pd.DataFrame]:
	if frameworks.get('matlab', False) or full:
		fcnet = preprocess_matlab(f'{results_root}/matlab_FullyConnectedNet.csv')
		scvnet = preprocess_matlab(f'{results_root}/matlab_SimpleConvNet.csv')
		mnet = preprocess_matlab(f'{results_root}/matlab_MobileNet-v2.csv')

	# training
	training_to_concat = []

	if frameworks.get('pytorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[pytorch.type == 'training']
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e9
		training_to_concat.append(pytorch)

	if frameworks.get('libtorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[(libtorch.type == 'training') | (libtorch.type == 'read')]
		libtorch.insert(0, 'framework', 'LibTorch')
		# kinda counterintuitive, but where changes values where the condition is false
		libtorch.elapsed_time.where(libtorch.model_name != 'CELEBA', libtorch.elapsed_time / 1e6, inplace=True)
		libtorch.elapsed_time.where(libtorch.model_name == 'CELEBA', libtorch.elapsed_time / 1e3, inplace=True)
		training_to_concat.append(libtorch)

	if frameworks.get('tensorflow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[tensorflow.type == 'training']
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e9
		training_to_concat.append(tensorflow)

	if frameworks.get('cudnn', False) or full:
		cudnn = preprocess_cudnn(results_root)
		cudnn = cudnn[cudnn.type == 'training']
		cudnn.insert(0, 'framework', 'cuDNN')
		cudnn.elapsed_time /= 1e3
		training_to_concat.append(cudnn)

	if frameworks.get('matlab', False) or full:
		matlab = pd.concat([fcnet, scvnet, mnet])
		matlab = matlab[matlab.type == 'training']
		matlab.insert(0, 'framework', 'Matlab')
		training_to_concat.append(matlab)

	training = pd.concat(training_to_concat)
	training.drop(columns=['type', 'loss', 'performance'], inplace=True)

	# evaluation
	evaluation_to_concat = []

	if frameworks.get('pytorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[pytorch.type != 'training']
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e6
		evaluation_to_concat.append(pytorch)

	if frameworks.get('libtorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[(libtorch.type != 'training') & (libtorch.type != 'read')]
		libtorch.insert(0, 'framework', 'LibTorch')
		evaluation_to_concat.append(libtorch)

	if frameworks.get('tensorflow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[tensorflow.type != 'training']
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e6
		evaluation_to_concat.append(tensorflow)

	if frameworks.get('cudnn', False) or full:
		cudnn = preprocess_cudnn(results_root)
		cudnn = cudnn[cudnn.type != 'training']
		cudnn.insert(0, 'framework', 'cuDNN')
		evaluation_to_concat.append(cudnn)

	if frameworks.get('matlab', False) or full:
		matlab = pd.concat([fcnet, scvnet, mnet])
		matlab = matlab[matlab.type != 'training']
		matlab.insert(0, 'framework', 'Matlab')
		matlab.elapsed_time *= 1e3
		evaluation_to_concat.append(matlab)

	evaluation = pd.concat(evaluation_to_concat)
	evaluation.drop(columns=['type', 'loss', 'performance', 'epoch'], inplace=True)
	evaluation.reset_index(drop=True, inplace=True)

	if save:
		idx = time.time_ns()
		training.to_csv(f'{results_root}/training_concat-{idx}.csv', index=False)
		evaluation.to_csv(f'{results_root}/evaluation_concat-{idx}.csv', index=False)

	return training, evaluation