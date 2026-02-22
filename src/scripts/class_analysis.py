import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
	# import the loader from the existing script in the same folder
	from randomforest import load_features
except Exception:
	load_features = None


def plot_class_imbalance(y, out_dir: Path):
	out_dir.mkdir(parents=True, exist_ok=True)

	counts = pd.Series(y).value_counts().sort_index()
	proportions = counts / counts.sum() * 100

	# Bar plot (counts)
	plt.figure(figsize=(10, 5))
	ax = sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis")
	ax.set_xlabel('Class')
	ax.set_ylabel('Count')
	ax.set_title('Class Counts')
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
	plt.tight_layout()
	plt.subplots_adjust(bottom=0.25)
	bar_path = out_dir / "class_counts_bar.png"
	plt.savefig(bar_path, bbox_inches='tight')
	plt.close()

	# Percent bar plot
	plt.figure(figsize=(10, 5))
	ax = sns.barplot(x=proportions.index.astype(str), y=proportions.values, palette="magma")
	ax.set_xlabel('Class')
	ax.set_ylabel('Percent (%)')
	ax.set_title('Class Distribution (%)')
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
	plt.tight_layout()
	plt.subplots_adjust(bottom=0.25)
	pct_path = out_dir / "class_counts_percent.png"
	plt.savefig(pct_path, bbox_inches='tight')
	plt.close()

	# Pie chart
	plt.figure(figsize=(6, 6))
	plt.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
	plt.title('Class Distribution (pie)')
	plt.tight_layout()
	pie_path = out_dir / "class_counts_pie.png"
	plt.savefig(pie_path)
	plt.close()

	return dict(bar=bar_path, percent=pct_path, pie=pie_path, counts=counts)


def main(features_dir: str, out_dir: str):
	features_path = Path(features_dir)
	out_path = Path(out_dir)

	if load_features is None:
		raise RuntimeError('Could not import load_features from randomforest.py; ensure scripts are runnable as a module.')

	X, y = load_features(features_path)

	results = plot_class_imbalance(y, out_path)

	print('Class counts:')
	print(results['counts'].to_string())
	print('\nSaved plots:')
	for k, p in results.items():
		if k == 'counts':
			continue
		print(f"- {k}: {p}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Analyze and plot class imbalance from extracted features')
	parser.add_argument('--features-dir', default=r"e:\\retinal-ai\\feature_extractions", help='Path to feature_extractions folder')
	parser.add_argument('--out-dir', default='analysis_outputs', help='Directory to save plots')
	args = parser.parse_args()

	main(args.features_dir, args.out_dir)

