# 🌐 English Language Learning Ability Prediction

---

## 📔 About

Welcome to our project, "English Language Learning Ability Prediction", an innovative venture by a team of dedicated data scientists from the Master of Data Science program at the University of British Columbia. The goal of our project was to forecast an individual's aptitude for learning by predicting their performance in an English speaking test, which is part of a quiz to assess English proficiency. This prediction is based on demographic details and their linguistic background, including factors such as the duration of their stay in English-speaking countries, their primary and native languages, among others. We aimed to develop a regression model utilizing both Ridge and Lasso models. The effectiveness of these models was evaluated using metrics like R-squared, Root Mean Squared Error (RMSE), and Negative Mean Squared Error (NMSE).
We utilized a subset of a dataset originally compiled from 680,333 participants, ranging in age from 7 to 89, for our study. These individuals completed an English grammar quiz and provided demographic information, as well as details about their language backgrounds and the countries they have lived in. This dataset was central to the research paper "A critical period for second language acquisition: Evidence from 2/3 million English speakers" by Joshua K. Hartshorne, Joshua B. Tenenbaum, and Steven Pinker. The study focused on understanding the peak age for grammar-learning ability, observing its maintenance during childhood, and its notable decline in late adolescence. The authors performed various analyses and compared multiple models to identify age-related patterns in language acquisition. The complete dataset is accessible at [https://osf.io/pyb8s/](https://osf.io/pyb8s/). For the initial phase of our project (Milestone 1), we only used 30% of the total dataset. The script for sampling this subset is available in 'src/Sampling from dataset.ipynb'. In future project stages, we plan to consider incorporating the entire dataset. This rich dataset allows for a multi-faceted analysis of language learning patterns.

---

## Report

The final report can be found
[here](https://ubc-mds.github.io/522-workflows-group-18/docs/english_language_learning_ability_prediction_analysis.html)

---

## 💻 Getting Started

### ⚙️ Initial Setup

1. Clone the repository and navigate to the project root

```bash
git clone
cd 522-workflows-group-18
```

2. Make sure Docker is installed and launched on your machine.

### 🛠️ Setting up your environment (conda or Docker)

#### Method 1: Running the code via conda environment

1. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate 522
```

2. Launch Jupyter Notebook:

```bash
jupyter lab
```

_Note_: If you want to close the environment, press `ctrl + c` or `cmd + c` in the terminal to exit Jupyter Notebook and run the following command to deactivate the environment:

```bash
conda deactivate 522
```

#### Method 2: Running the code via Docker container

1. Go to the project root and run the following command:

```bash
docker compose up # add -d flag to run in detached mode
```

_Link to Docker Hub image:_ [farrandi/522-workflows-group-18](https://hub.docker.com/r/farrandi/522-workflows-group-18)

2. Click the link in the terminal to launch Jupyter Notebook. It should look something like this: `http://127.0.0.1:8888/lab`

3. Navigate to the analysis notebook at `work/notebooks/english_language_learning_ability_prediction_analysis.ipynb`

_Note_: when you want toto close the container, press `ctrl + c` or `cmd + c` in the terminal and run the following command:

```bash
docker compose down
```

### 🖥️ Viewing the report

1. Make sure you have done the steps in the previous section and are in the jupyter lab UI (either via the `jupyter lab` command from the conda environment or via the link in the terminal when you run `docker compose up`)

2. Navigate to `notebooks/english_language_learning_ability_prediction_analysis.ipynb` to view our data analysis process, model training, and predictions.

3. Feel free to click `Restart Kernel and Run All Cells` to re-run the analysis.

### 🏃 Running the analysis

**Note:** Currently this functionality only works via the conda environment.

1. Make sure you followed Method 1 in the **Setting up your environment** section.

2. Navigate to the project root and run the following command.

These are the simple commands with all the default values to run the analysis

```bash
# Get train/test data:
python src/scripts/english_score_get_data.py

# Perform EDA:
python src/scripts/english_score_eda.py -v

# Tune Models:
python src/scripts/english_score_tuning.py -v

# Get Optimal Model Results:
python src/scripts/english_score_results.py -v
```

or you can run the following commands to customize the analysis:

```bash
# Get train/test data:
python src/scripts/english_score_get_data.py\
   --url="https://osf.io/download/g72pq/" \
   --output_folder_path="./data/raw"

# Perform EDA:
python src/scripts/english_score_eda.py -v \
   --training-data="data/raw/train_data.csv" \
   --plot-to="results/figures/" \
   --pickle-to="results/models/preprocessor/" \
   --tables-to="results/tables/"

# Tune Models:
python src/scripts/english_score_tuning.py -v \
    --train="data/raw/train_data.csv" \
    --test="data/raw/test_data.csv" \
    --output_dir="results/" \
    --preprocessor_path="results/models/preprocessor/preprocessor.pkl"

# Get Optimal Model Results:
python src/scripts/english_score_results.py -v \
    --train="data/raw/train_data.csv" \
    --test="data/raw/test_data.csv" \
    --plot-to="results/figures/" \
    --tables-to="results/tables/" \
    --preprocessor_path="results/models/preprocessor/preprocessor.pkl"
    --best_model_path="results/models/ridge_best_model.pkl"

# Build HTML report and copy build to docs folder
jupyter-book build notebooks
cp -r notebooks/_build/html/ docs
```

---

## 🔍 Methodology

We employ a comprehensive approach:

1. **Data Preprocessing:** Cleaning, normalization, and transformation of data to ensure quality and consistency.
2. **Exploratory Data Analysis:** Utilizing statistical techniques and visualization tools to uncover trends and patterns.
3. **Model Development:** We developed predictive models by employing Ridge and Lasso regression techniques to ascertain the weight of each variable in predicting English language proficiency. By fine-tuning these models, we aimed to minimize prediction errors and enhance the accuracy of our forecasts.
4. **Model Evaluation:** Our models were rigorously evaluated using cross-validation techniques and a variety of performance metrics such as R-squared, Root Mean Squared Error (RMSE), ensuring their robustness and reliability in predicting language learning ability.

---

## 📈 Results and Discussion

Our analysis, based on a dataset of approximately 200,000 individuals, shows that factors such as age, education, and language background significantly predict English proficiency. The regression model achieved a 5.3% RMSE on test data, confirming the reliability of these predictors in assessing language skills. Notably, being a native English speaker emerged as the strongest positive predictor, while immersion in English learning showed a strong negative correlation. These findings reinforce that demographic and educational backgrounds are crucial in language acquisition. This opens avenues for future research, particularly in understanding how cultural exposure influences language proficiency, potentially leading to more effective and personalized language learning strategies.

---

## 👥 Team Members

- Salva
- Atabak
- Nando
- Rachel

---

## 🤝 Contributing

Your contributions can help enhance this project further. For contribution guidelines, please refer to `CONTRIBUTING.md`. We appreciate your interest in improving the predictive capabilities of our model.

## ©️ License

This project is released under the MIT License, promoting open-source collaboration. For more details, see `LICENSE.md`.

## 🙏 Acknowledgments

Special thanks to our course instructors and UBC for providing the resources and support necessary for this project.

## ✉️ Contact

For queries or collaborations, feel free to contact any of our team members.
