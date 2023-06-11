import pandas as pd

def preprocess_matlab(filepath: str):
	mname = filepath[filepath.index('_')+1:filepath.index('.')]
	matlab = pd.read_csv(filepath, usecols=('Epoch', 'TimeSinceStart', 'TrainingLoss', 'ValidationAccuracy', 'State'))
	matlab = matlab.rename(columns=dict(zip(matlab.columns, ['eps','times','loss','acc','type'])))

	matlab.insert(0, 'type', matlab.pop('type'))
	matlab.insert(4, 'times', matlab.pop('times'))

	matlab.loss = matlab.loss.fillna(-1)
	matlab = matlab.dropna(axis=0)

	matlab.insert(0, 'framework', 'MATLAB')
	matlab.insert(1, 'mnames', mname)

	matlab.times *= 1e3
	matlab = matlab[matlab.type != 'done'].drop([1])
	matlab = matlab.reset_index().drop('index', axis=1)
	matlab.type = matlab.type.replace('iteration', 'training')
	matlab.loc[1:14, 'times'] = matlab.times[1:-1] - matlab.times.shift(1)[1:-1]

	return matlab

cudnn_fcnet = pd.read_csv('../results/cpp_fcnet.csv')
cudnn_fcnet.insert(0, 'framework', 'cuDNN')
cudnn_fcnet.acc /= 100

cudnn_scvnet = pd.read_csv('../results/cpp_scvnet.csv')
cudnn_scvnet.insert(0, 'framework', 'cuDNN')
cudnn_scvnet.acc /= 100

matlab_fcnet = preprocess_matlab('../results/matlab_fcnet.csv')
matlab_mobilenet = preprocess_matlab('../results/matlab_mobilenet_v2.csv')

pytorch = pd.read_csv('../results/pytorch_results.csv')
pytorch.insert(0, 'framework', 'PyTorch')

tensorflow = pd.read_csv('../results/tensorflow_results.csv')
tensorflow.insert(0, 'framework', 'TensorFlow')
tensorflow.times /= 1e6

results_concat = pd.concat((cudnn_fcnet, cudnn_scvnet, matlab_fcnet, matlab_mobilenet, pytorch, tensorflow), axis=0)
results_concat = results_concat.rename(columns={'mnames': 'model_name', 'eps': 'epochs', 'acc': 'accuracy', 'times': 'exec_time'})
results_concat.to_csv('results_concat.csv', index=False)