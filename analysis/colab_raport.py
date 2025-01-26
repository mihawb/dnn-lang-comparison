import pandas as pd
from pathlib import Path


colab_results_root = Path("../results_colab")


def group_by_timestamp(results: list[Path]) -> dict[str, list[Path]]:
	timestamps = set([f.name[f.name.rfind("-")+1:-4] for f in results])
	return {t: [f for f in results if f.name[f.name.rfind("-")+1:-4] == t] for t in timestamps}


def validate_result_pair(results: list[Path]) -> None:
	timestamp = results[0].name[results[0].name.rfind("-")+1:-4]
	if len(results) < 2:
		print(f"Skipping {timestamp} - not enough files for this timestamp\n\n")
		return
	
	if len(results) > 2:
		print(f"somehow more than two files for this timestamp: {timestamp}\n\n")
		return
	
	# w kazdym pliku powinno wystapic N epochow (N = 8 raczej) dla kazdego modelu
	res1 = pd.read_csv(results[0])
	print(f"Epoch count by model for {results[0].name}:")
	print(res1[["model_name", "phase"]].value_counts())

	res2 = pd.read_csv(results[1])
	print(f"Epoch count by model for {results[1].name}:")
	print(res2.model_name.value_counts())
	



results = [f for f in colab_results_root.glob("*.csv")]
results_grouped = group_by_timestamp(results)

for timestamp, results in results_grouped.items():
	validate_result_pair(results)
	print("\n\n")
