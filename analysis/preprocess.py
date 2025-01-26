import pandas as pd
import time


def preprocess_matlab(filepath: str) -> pd.DataFrame:
	mname = filepath[filepath.rindex('_')+1:filepath.rindex('.')]
	matlab = pd.read_csv(filepath, usecols=('Epoch', 'TimeSinceStart', 'TrainingLoss', 'ValidationAccuracy', 'State'))
	matlab = matlab.rename(columns=dict(zip(matlab.columns, ['epoch','elapsed_time','loss','performance','phase'])))

	matlab.insert(0, 'phase', matlab.pop('phase'))
	matlab.insert(4, 'elapsed_time', matlab.pop('elapsed_time'))

	matlab.loss = matlab.loss.fillna(-1)
	matlab = matlab.dropna(axis=0)
	matlab.insert(0, 'model_name', mname)

	# matlab.elapsed_time *= 1e3
	matlab = matlab[matlab.phase != 'done'].drop([1])
	matlab = matlab.reset_index().drop('index', axis=1)
	matlab.phase = matlab.phase.replace('iteration', 'training')
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
		rnet = preprocess_matlab(f'{results_root}/matlab_ResNet-50.csv')
		dcgan = pd.read_csv(f'{results_root}/matlab_DCGAN.csv')

	# training
	training_to_concat = []

	if frameworks.get('pytorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[pytorch.phase == 'training']
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e9
		training_to_concat.append(pytorch)

	if frameworks.get('libtorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[libtorch.phase == 'training']
		libtorch.insert(0, 'framework', 'LibTorch')
		# kinda counterintuitive, but where changes values where the condition is false
		libtorch.elapsed_time.where(libtorch.model_name != 'CELEBA', libtorch.elapsed_time / 1e6, inplace=True)
		libtorch.elapsed_time.where(libtorch.model_name == 'CELEBA', libtorch.elapsed_time / 1e3, inplace=True)
		training_to_concat.append(libtorch)

	if frameworks.get('tensorflow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[tensorflow.phase == 'training']
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e9
		training_to_concat.append(tensorflow)

	if frameworks.get('cudnn', False) or full:
		cudnn = preprocess_cudnn(results_root)
		cudnn = cudnn[cudnn.phase == 'training']
		cudnn.insert(0, 'framework', 'cuDNN')
		cudnn.elapsed_time /= 1e3
		training_to_concat.append(cudnn)

	if frameworks.get('matlab', False) or full:
		matlab = pd.concat([fcnet, scvnet, mnet, rnet, dcgan])
		matlab = matlab[matlab.phase == 'training']
		matlab.insert(0, 'framework', 'Matlab')
		training_to_concat.append(matlab)

	training = pd.concat(training_to_concat)
	training.drop(columns=['phase', 'loss', 'performance'], inplace=True)

	# evaluation
	evaluation_to_concat = []

	if frameworks.get('pytorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[(pytorch.phase != 'training') & (pytorch.phase != 'latency')]
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e6
		evaluation_to_concat.append(pytorch)

	if frameworks.get('libtorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[(libtorch.phase != 'training') & (libtorch.phase != 'latency')]
		libtorch.insert(0, 'framework', 'LibTorch')
		evaluation_to_concat.append(libtorch)

	if frameworks.get('tensorflow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[(tensorflow.phase != 'training') & (tensorflow.phase != 'latency')]
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e6
		evaluation_to_concat.append(tensorflow)

	if frameworks.get('cudnn', False) or full:
		cudnn = preprocess_cudnn(results_root)
		cudnn = cudnn[(cudnn.phase != 'training') & (cudnn.phase != 'latency')]
		cudnn.insert(0, 'framework', 'cuDNN')
		evaluation_to_concat.append(cudnn)

	if frameworks.get('matlab', False) or full:
		matlab = pd.concat([fcnet, scvnet, mnet, rnet, dcgan])
		matlab = matlab[(matlab.phase != 'training') & (matlab.phase != 'latency')]
		matlab.insert(0, 'framework', 'Matlab')
		matlab.elapsed_time *= 1e3
		evaluation_to_concat.append(matlab)

	evaluation = pd.concat(evaluation_to_concat)
	evaluation.drop(columns=['phase', 'loss', 'performance', 'epoch'], inplace=True)
	evaluation.reset_index(drop=True, inplace=True)

	# latency
	latency_to_concat = []

	if frameworks.get('pytorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[pytorch.phase == 'latency']
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e6
		latency_to_concat.append(pytorch)

	if frameworks.get('libtorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[libtorch.phase == 'latency']
		libtorch.insert(0, 'framework', 'LibTorch')
		latency_to_concat.append(libtorch)

	if frameworks.get('tensorflow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[tensorflow.phase == 'latency']
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e6
		latency_to_concat.append(tensorflow)

	if frameworks.get('cudnn', False) or full:
		cudnn = preprocess_cudnn(results_root)
		cudnn = cudnn[cudnn.phase == 'latency']
		cudnn.insert(0, 'framework', 'cuDNN')
		latency_to_concat.append(cudnn)

	if frameworks.get('matlab', False) or full:
		matlab = pd.concat([fcnet, scvnet, mnet, rnet, dcgan])
		matlab = matlab[matlab.phase == 'latency']
		matlab.insert(0, 'framework', 'Matlab')
		matlab.elapsed_time *= 1e3
		latency_to_concat.append(matlab)

	latency = pd.concat(latency_to_concat)
	latency.drop(columns=['phase', 'loss', 'performance'], inplace=True)
	latency.reset_index(drop=True, inplace=True)

	if save:
		idx = time.time_ns()
		training.to_csv(f'{results_root}/training_concat-{idx}.csv', index=False)
		evaluation.to_csv(f'{results_root}/evaluation_concat-{idx}.csv', index=False)
		latency.to_csv(f'{results_root}/latenct_concat-{idx}.csv', index=False)

	return training, evaluation, latency


def preprocess_cudnn_mod(results_root: str) -> pd.DataFrame:
	fcnet = pd.read_csv(f'{results_root}/cudnn_fcnet.csv')
	fcnet.model_name = 'FullyConnectedNet'

	return fcnet


def mean_aggregate_results(ts: list[pd.DataFrame], es: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
	for t, e in zip(ts, es):
		t.set_index(['framework', 'model_name', 'epoch'], inplace=True)
		e.set_index(['framework', 'model_name'], inplace=True)

	res = []
	for rs in [ts, es]:
		acc_rs = rs[0].copy()
		for i, r in enumerate(rs[1:], 1):
			acc_rs = acc_rs.merge(r, left_index=True, right_index=True, suffixes=('', f'_{i}'))

		mean_rs = acc_rs.mean(axis=1)
		mean_rs = mean_rs.reset_index()
		mean_rs.rename(columns={0: 'elapsed_time'}, inplace=True)
		res.append(mean_rs)

	return tuple(res)


def get_results_mod(results_root: str, save: bool=False, full: bool=True, **frameworks) -> tuple[pd.DataFrame, pd.DataFrame]:
	if frameworks.get('matlab', False) or full:
		fcnet = preprocess_matlab(f'{results_root}/matlab_FullyConnectedNet.csv')

	# training
	training_to_concat = []

	if frameworks.get('pytorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[pytorch.phase == 'training']
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e9
		training_to_concat.append(pytorch)

	if frameworks.get('libtorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[libtorch.phase == 'training']
		libtorch.insert(0, 'framework', 'LibTorch')
		# kinda counterintuitive, but where changes values where the condition is false
		libtorch.elapsed_time.where(libtorch.model_name != 'CELEBA', libtorch.elapsed_time / 1e6, inplace=True)
		libtorch.elapsed_time.where(libtorch.model_name == 'CELEBA', libtorch.elapsed_time / 1e3, inplace=True)
		training_to_concat.append(libtorch)

	if frameworks.get('tensorflow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[tensorflow.phase == 'training']
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e9
		training_to_concat.append(tensorflow)

	if frameworks.get('cudnn', False) or full:
		cudnn = preprocess_cudnn_mod(results_root)
		cudnn = cudnn[cudnn.phase == 'training']
		cudnn.insert(0, 'framework', 'cuDNN')
		cudnn.elapsed_time /= 1e3
		training_to_concat.append(cudnn)

	if frameworks.get('matlab', False) or full:
		matlab = fcnet.copy()
		matlab = matlab[matlab.phase == 'training']
		matlab.insert(0, 'framework', 'Matlab')
		training_to_concat.append(matlab)

	training = pd.concat(training_to_concat)
	training.drop(columns=['phase', 'loss', 'performance'], inplace=True)

	# evaluation
	evaluation_to_concat = []

	if frameworks.get('pytorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[pytorch.phase != 'training']
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e6
		evaluation_to_concat.append(pytorch)

	if frameworks.get('libtorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[libtorch.phase != 'training']
		libtorch.insert(0, 'framework', 'LibTorch')
		evaluation_to_concat.append(libtorch)

	if frameworks.get('tensorflow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[tensorflow.phase != 'training']
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e6
		evaluation_to_concat.append(tensorflow)

	if frameworks.get('cudnn', False) or full:
		cudnn = preprocess_cudnn_mod(results_root)
		cudnn = cudnn[cudnn.phase != 'training']
		cudnn.insert(0, 'framework', 'cuDNN')
		evaluation_to_concat.append(cudnn)

	if frameworks.get('matlab', False) or full:
		matlab = fcnet.copy()
		matlab = matlab[matlab.phase != 'training']
		matlab.insert(0, 'framework', 'Matlab')
		matlab.elapsed_time *= 1e3
		evaluation_to_concat.append(matlab)

	evaluation = pd.concat(evaluation_to_concat)
	evaluation.drop(columns=['phase', 'loss', 'performance', 'epoch'], inplace=True)
	evaluation.reset_index(drop=True, inplace=True)

	if save:
		idx = time.time_ns()
		training.to_csv(f'{results_root}/training_concat-{idx}.csv', index=False)
		evaluation.to_csv(f'{results_root}/evaluation_concat-{idx}.csv', index=False)

	return training, evaluation