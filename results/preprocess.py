import pandas as pd

def preprocess_matlab(filepath):
	matlab = pd.read_csv(filepath, usecols=('Epoch', 'TimeSinceStart', 'TrainingLoss', 'ValidationAccuracy', 'State'))
	matlab = matlab.rename(columns=dict(zip(matlab.columns, ['eps','times','loss','acc','type'])))

	matlab.insert(0, 'type', matlab.pop('type'))
	matlab.insert(4, 'times', matlab.pop('times'))

	matlab.loss = matlab.loss.fillna(-1)
	matlab = matlab.dropna(axis=0)

	matlab.insert(0, 'framework', 'matlab')
	matlab.insert(1, 'mnames', 'fcnet')

	matlab.times *= 1e3
	matlab = matlab[matlab.type != 'done'].drop([1])
	matlab = matlab.reset_index().drop('index', axis=1)
	matlab.type = matlab.type.replace('iteration', 'training')
	matlab.loc[1:14, 'times'] = matlab.times[1:-1] - matlab.times.shift(1)[1:-1]

	return matlab

matlab_fcnet = preprocess_matlab('matlab-fcnet.csv')
matlab_mobilenet = preprocess_matlab('matlab-mobilenet-v2.csv')

pytorch = pd.read_csv('pytorch_results.csv')
pytorch.insert(0, 'framework', 'pytorch')

tensorflow = pd.read_csv('tensorflow_results.csv')
tensorflow.insert(0, 'framework', 'tensorflow')
tensorflow.times /= 1e6

cudnn_fcnet = pd.read_csv('cpp_fcnet.csv')
cudnn_fcnet.insert(0, 'framework', 'cudnn')

cudnn_scvnet = pd.read_csv('cpp_scvnet.csv')
cudnn_scvnet.insert(0, 'framework', 'cudnn')

results_concat = pd.concat((cudnn_fcnet, cudnn_scvnet, matlab_fcnet, matlab_mobilenet, pytorch, tensorflow), axis=0)
results_concat.to_csv('results_concat.csv', index=False)