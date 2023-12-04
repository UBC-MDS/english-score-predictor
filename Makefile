# Run the scripts
all: docs/index.html

# Target for getting train/test data
data/raw/train_data.csv data/raw/test_data.csv data/raw/sampled_data.zip:
	python src/scripts/english_score_get_data.py \
	--url="https://osf.io/download/g72pq/" \
	--output_folder_path="./data/raw"

# Target for performing EDA
results/models/preprocessor/preprocessor.pkl: data/raw/train_data.csv
	python src/scripts/english_score_eda.py -v \
	--training-data="data/raw/train_data.csv" \
	--plot-to="results/figures/" \
	--pickle-to="results/models/preprocessor/" \
	--tables-to="results/tables/"

# Target for tuning models
results/models/ridge_best_model.pkl: data/raw/train_data.csv data/raw/test_data.csv results/models/preprocessor/preprocessor.pkl
	python src/scripts/english_score_tuning.py -v \
	--train="data/raw/train_data.csv" \
	--test="data/raw/test_data.csv" \
	--output_dir="results/" \
	--preprocessor_path="results/models/preprocessor/preprocessor.pkl"

# Target for getting optimal model results
figures/act-vs-pred.png tables/test-score.csv: data/raw/train_data.csv data/raw/test_data.csv results/models/ridge_best_model.pkl
	python src/scripts/english_score_results.py -v \
	--train="data/raw/train_data.csv" \
	--test="data/raw/test_data.csv" \
	--plot_to="results/figures/" \
	--tables_to="results/tables/" \
	--preprocessor_path="results/models/preprocessor/preprocessor.pkl" \
	--best_model_path="results/models/ridge_best_model.pkl"

# Build the docs
notebooks/_build/html/docs/index.html: figures/act-vs-pred.png tables/test-score.csv
	jupyter-book build notebooks

# move the docs to docs
docs/index.html: notebooks/_build/html/docs/index.html
	cp -r notebooks/_build/html/docs

# Clean up everything
clean:
	rm -r data/raw/*.csv
	rm -r data/raw/*.zip
	rm -r results/figures/*.png
	rm -r results/tables/*.csv
	rm -r results/models/*.pkl
	rm -r results/models/preprocessor/*.pkl
	rm -rf notebooks/_build
	rm -rf docs
