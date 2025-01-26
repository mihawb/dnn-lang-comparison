import pandas as pd
import time
from pathlib import Path
from functools import reduce


MODEL_ORDER = ["FullyConnectedNet", "SimpleConvNet", "ResNet-50", "DenseNet-121", "MobileNet-v2", "ConvNeXt-Tiny", "DCGAN", "YOLOv8m"]
FRAMEWORK_ORDER = ["cuDNN", "LibTorch", "PyTorch", "TensorFlow", "Matlab", "PyTorch_eager", "PyTorch_compile"]


def make_key(order: list[str]) -> callable:
	d = {x: i for i, x in enumerate(order)}
	def key(s: pd.Series) -> pd.Series:
		return s.map(d)
	return key


def preprocess_matlab(filepath: str) -> pd.DataFrame:
	mname = filepath[filepath.rindex('_')+1:filepath.rindex('.')]
	matlab = pd.read_csv(filepath, usecols=('Epoch', 'TimeSinceStart', 'TrainingLoss', 'ValidationAccuracy', 'State'))
	matlab = matlab.rename(columns=dict(zip(matlab.columns, ['epoch','elapsed_time','loss','performance','phase'])))

	matlab.insert(0, 'phase', matlab.pop('phase'))
	matlab.insert(4, 'elapsed_time', matlab.pop('elapsed_time'))

	matlab.loss = matlab.loss.fillna(-1)
	matlab = matlab.dropna(axis=0)
	assert "latency" in matlab.phase.values, f"(3) missing latency phase in {mname}"
	matlab.insert(0, 'model_name', mname)

	# matlab.elapsed_time *= 1e3
	matlab = matlab[matlab.phase != 'done'].drop([1])
	matlab = matlab.reset_index(drop=True)  #.drop('index', axis=1)
	matlab.phase = matlab.phase.replace('iteration', 'training')
	matlab.epoch = matlab.epoch.apply(lambda ep: ep - 1000 if ep > 1000 else ep)
	matlab.loc[1:max(matlab[matlab.phase == "training"].epoch)-1, 'elapsed_time'] = matlab.elapsed_time[1:-1] - matlab.elapsed_time.shift(1)[1:-1]

	return matlab


def preprocess_cudnn(results_root: str) -> pd.DataFrame:
	fcnet = pd.read_csv(f'{results_root}/cudnn_fcnet.csv')
	fcnet.model_name = 'FullyConnectedNet'

	scvnet = pd.read_csv(f'{results_root}/cudnn_scvnet.csv')
	scvnet.model_name = 'SimpleConvNet'

	return pd.concat([fcnet, scvnet])


def preprocess_pytorch_compile(results_root: str) -> pd.DataFrame:
	results_root: Path = Path(results_root)

	clf_eager = pd.read_csv(n := next(results_root.glob('pytorch-clf-eager-*.csv')))
	assert (s := clf_eager.shape[0]) == 228, f"results inconsistency ({s} rows) in {n.resolve()}"

	clf_compile = pd.read_csv(n := next(results_root.glob('pytorch-clf-compile-*.csv')))
	assert (s := clf_compile.shape[0]) == 228, f"results inconsistency ({s} rows) in {n.resolve()}"

	dcgan_eager = pd.read_csv(n := next(results_root.glob('pytorch-dcgan-eager-*.csv')))
	assert (s := dcgan_eager.shape[0]) == 38, f"results inconsistency ({s} rows) in {n.resolve()}"

	dcgan_compile = pd.read_csv(n := next(results_root.glob('pytorch-dcgan-compile-*.csv')))
	assert (s := dcgan_compile.shape[0]) == 38, f"results inconsistency ({s} rows) in {n.resolve()}"

	clf_eager.insert(0, 'framework', 'PyTorch_eager')
	clf_compile.insert(0, 'framework', 'PyTorch_compile')
	dcgan_eager.insert(0, 'framework', 'PyTorch_eager')
	dcgan_compile.insert(0, 'framework', 'PyTorch_compile')

	concat = pd.concat([clf_eager, clf_compile, dcgan_eager, dcgan_compile])
	return concat


def get_results(results_root: str, frameworks: dict, save: bool=False, full: bool=True, train_preserve_warmup: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if frameworks.get('Matlab', False) or full:
		fcnet = preprocess_matlab(f'{results_root}/matlab_FullyConnectedNet.csv')
		scvnet = preprocess_matlab(f'{results_root}/matlab_SimpleConvNet.csv')
		mnet = preprocess_matlab(f'{results_root}/matlab_MobileNet-v2.csv')
		rnet = preprocess_matlab(f'{results_root}/matlab_ResNet-50.csv')
		dcgan = pd.read_csv(f'{results_root}/matlab_DCGAN.csv')

	# training
	training_to_concat = []

	if frameworks.get('PyTorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[pytorch.phase == 'training']
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e9
		training_to_concat.append(pytorch)

	if frameworks.get('PyTorch_compile', False) or full:
		pytorch_compile = preprocess_pytorch_compile(results_root)
		pytorch_compile = pytorch_compile[pytorch_compile.phase == 'training']
		# framework already added in the preprocessing
		pytorch_compile.elapsed_time /= 1e3
		training_to_concat.append(pytorch_compile)

	if frameworks.get('LibTorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[libtorch.phase == 'training']
		libtorch.insert(0, 'framework', 'LibTorch')
		libtorch = libtorch[libtorch.model_name != "CELEBA"].reset_index(drop=True)
		libtorch.elapsed_time /= 1e3
		training_to_concat.append(libtorch)

	if frameworks.get('TensorFlow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[tensorflow.phase == 'training']
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e9
		training_to_concat.append(tensorflow)

	if frameworks.get('cuDNN', False) or full:
		cudnn = preprocess_cudnn(results_root)
		cudnn = cudnn[cudnn.phase == 'training']
		cudnn.insert(0, 'framework', 'cuDNN')
		cudnn.elapsed_time /= 1e3
		training_to_concat.append(cudnn)

	if frameworks.get('Matlab', False) or full:
		matlab = pd.concat([fcnet, scvnet, mnet, rnet, dcgan])
		matlab = matlab[matlab.phase == 'training']
		matlab.insert(0, 'framework', 'Matlab')
		training_to_concat.append(matlab)

	training = pd.concat(training_to_concat)
	if not train_preserve_warmup:
		training = training[training.epoch > 2]
	training.drop(columns=['phase', 'loss', 'performance'], inplace=True)

	# evaluation
	evaluation_to_concat = []

	if frameworks.get('PyTorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[(pytorch.phase != 'training') & (pytorch.phase != 'latency')]
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e6
		evaluation_to_concat.append(pytorch)

	if frameworks.get('PyTorch_compile', False) or full:
		# eval for torch_compile used for graph compilations and batch latency
		pytorch_compile = preprocess_pytorch_compile(results_root)
		pytorch_compile = pytorch_compile[(pytorch_compile.phase == 'graph_compilation') | (pytorch_compile.phase == 'graph_compilation_batch')]
		# framework already added in the preprocessing
		# leaving elapsed_time in ms
		evaluation_to_concat.append(pytorch_compile)

	if frameworks.get('LibTorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[(libtorch.phase != 'training') & (libtorch.phase != 'latency')]
		libtorch.insert(0, 'framework', 'LibTorch')
		evaluation_to_concat.append(libtorch)

	if frameworks.get('TensorFlow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[(tensorflow.phase != 'training') & (tensorflow.phase != 'latency')]
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e6
		evaluation_to_concat.append(tensorflow)

	if frameworks.get('cuDNN', False) or full:
		cudnn = preprocess_cudnn(results_root)
		cudnn = cudnn[(cudnn.phase != 'training') & (cudnn.phase != 'latency')]
		cudnn.insert(0, 'framework', 'cuDNN')
		evaluation_to_concat.append(cudnn)

	if frameworks.get('Matlab', False) or full:
		matlab = pd.concat([fcnet, scvnet, mnet, rnet, dcgan])
		matlab = matlab[(matlab.phase != 'training') & (matlab.phase != 'latency')]
		matlab.insert(0, 'framework', 'Matlab')
		matlab.elapsed_time *= 1e3
		evaluation_to_concat.append(matlab)

	evaluation = pd.concat(evaluation_to_concat)
	evaluation.drop(columns=['loss', 'performance', 'epoch'], inplace=True)
	evaluation.reset_index(drop=True, inplace=True)

	# latency
	latency_to_concat = []

	if frameworks.get('PyTorch', False) or full:
		pytorch = pd.read_csv(f'{results_root}/pytorch.csv')
		pytorch = pytorch[pytorch.phase == 'latency']
		pytorch.insert(0, 'framework', 'PyTorch')
		pytorch.elapsed_time /= 1e6
		latency_to_concat.append(pytorch)

	if frameworks.get('PyTorch_compile', False) or full:
		pytorch_compile = preprocess_pytorch_compile(results_root)
		pytorch_compile = pytorch_compile[(pytorch_compile.phase == 'latency') | (pytorch_compile.phase == 'latency_batch')]
		# framework already added in the preprocessing
		# leaving elapsed_time in ms
		latency_to_concat.append(pytorch_compile)

	if frameworks.get('LibTorch', False) or full:
		libtorch = pd.read_csv(f'{results_root}/libtorch.csv')
		libtorch = libtorch[libtorch.phase == 'latency']
		libtorch.insert(0, 'framework', 'LibTorch')
		latency_to_concat.append(libtorch)

	if frameworks.get('TensorFlow', False) or full:
		tensorflow = pd.read_csv(f'{results_root}/tensorflow.csv')
		tensorflow = tensorflow[tensorflow.phase == 'latency']
		tensorflow.insert(0, 'framework', 'TensorFlow')
		tensorflow.elapsed_time /= 1e6
		latency_to_concat.append(tensorflow)

	if frameworks.get('cuDNN', False) or full:
		cudnn = preprocess_cudnn(results_root)
		cudnn = cudnn[cudnn.phase == 'latency']
		cudnn.insert(0, 'framework', 'cuDNN')
		latency_to_concat.append(cudnn)

	if frameworks.get('Matlab', False) or full:
		matlab = pd.concat([fcnet, scvnet, mnet, rnet, dcgan])
		matlab = matlab[matlab.phase == 'latency']
		matlab.insert(0, 'framework', 'Matlab')
		matlab.elapsed_time *= 1e3
		latency_to_concat.append(matlab)

	latency = pd.concat(latency_to_concat)
	if not train_preserve_warmup:
		latency = latency[latency.epoch > 2]
	latency.drop(columns=['loss', 'performance'], inplace=True)
	latency.reset_index(drop=True, inplace=True)

	training = training[training.model_name != "SODNet"].reset_index(drop=True)
	evaluation = evaluation[evaluation.model_name != "SODNet"].reset_index(drop=True)
	latency = latency[latency.model_name != "SODNet"].reset_index(drop=True)

	if save:
		idx = time.time_ns()
		training.to_csv(f'{results_root}/training_concat-{idx}.csv', index=False)
		evaluation.to_csv(f'{results_root}/evaluation_concat-{idx}.csv', index=False)
		latency.to_csv(f'{results_root}/latenct_concat-{idx}.csv', index=False)

	return training, evaluation, latency


def mean_aggregate_results(
	ts: pd.DataFrame, es: pd.DataFrame, ls: pd.DataFrame, 
	train_preserve_epoch: bool = False,
	latency_preserve_epoch: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Input is the output from get_concat_results"""
	indexes = [
		['framework', 'model_name', 'epoch'] if train_preserve_epoch else ['framework', 'model_name'],
		['framework', 'model_name', 'phase'],
		['framework', 'model_name', 'phase', 'epoch'] if latency_preserve_epoch else ['framework', 'model_name', 'phase']
	]

	if not train_preserve_epoch:
		ts = ts.drop(["epoch"], axis=1)
	
	if not latency_preserve_epoch:
		ls = ls.drop(["epoch"], axis=1)

	res = []
	for i, rs in enumerate([ts, es, ls]):
		g = rs.groupby(indexes[i])
		mean_rs = g.mean().rename(columns=lambda c: c + "_mean")
		median_rs = g.median().rename(columns=lambda c: c + "_median")
		std_rs = g.std().rename(columns=lambda c: c + "_std")
		all_rs = pd.concat([mean_rs, median_rs, std_rs], axis="columns")
		res.append(all_rs)

	for r in res:
		r.reset_index(inplace=True)
		r.sort_values(by=["framework", "model_name"], kind="stable", 
				key=make_key(FRAMEWORK_ORDER + MODEL_ORDER), inplace=True)
		r.reset_index(inplace=True, drop=True)

	return tuple(res)


def get_concat_results(file_name_format: str, r_start: int, r_end: int, train_preserve_warmup: bool, frameworks: dict = dict(), full: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	ts,	es,	ls = [], [], []

	for i in range(r_start, r_end + 1):
		t, e, l = get_results(file_name_format % i, frameworks, full=full, train_preserve_warmup=train_preserve_warmup) 
		ts.append(t)
		es.append(e)
		ls.append(l)

	return [reduce(lambda agg, new: pd.concat([agg, new]), dfs[1:], dfs[0]) for dfs in [ts, es, ls]]


def set_vertical_grid(ax):
	ax.grid(axis="y", which="major", zorder=1)
	ax.minorticks_on()
	ax.tick_params(axis='x', which='minor', bottom=False)
	ax.grid(visible=True, axis="y", which="minor", alpha=0.2, zorder=1)
	ax.legend().set_visible(False)
