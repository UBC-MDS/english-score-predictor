# 🌐 English Language Learning Ability Prediction

---

<img src="images/language-learning.jpg" width="700">

## 📔 About

Welcome to our project, **"English Language Learning Ability Prediction"**, an innovative venture by some students from the Master of Data Science program at the University of British Columbia.

The goal of our project was to forecast an individual's aptitude for learning by **predicting their performance in an English speaking test**, which is part of a quiz to assess English proficiency. This prediction is based on demographic details and their linguistic background, including factors such as the duration of their stay in English-speaking countries, their primary and native languages, among others. We aimed to develop a regression model utilizing both Ridge and Lasso models. The effectiveness of these models was evaluated using metrics like R-squared, Root Mean Squared Error (RMSE), and Negative Mean Squared Error (NMSE).

We utilized a subset of a dataset originally compiled from 680,333 participants, ranging in age from 7 to 89, for our study. These individuals completed an English grammar quiz and provided demographic information, as well as details about their language backgrounds and the countries they have lived in.

The complete dataset is accessible at [https://osf.io/pyb8s/](https://osf.io/pyb8s/). For the purpose of this analysis, we only used 30% of the total dataset. The script for sampling this subset is available in `src/scripts/english_score_get_data.py`. In future project stages, we plan to consider incorporating the entire dataset. This rich dataset allows for a multi-faceted analysis of language learning patterns.

---

## Report

The final report can be found [here](https://ubc-mds.github.io/522-workflows-group-18/docs/english_language_learning_ability_prediction_analysis.html).

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

- Now you are in the terminal with the conda environment activated. You can confirm this if you see `(522)` at the beginning of the terminal line.

2. Launch Jupyter Notebook:

```bash
jupyter lab
```

_Note_: If you want to close the environment, press `ctrl + c` or `cmd + c` in the terminal to exit Jupyter Notebook and run the following command to deactivate the environment:

```bash
conda deactivate 522
```

#### Method 2: Running the code via Docker container (Recommended)

1. Go to the project root and run the following command:

```bash
docker compose up # add -d flag to run in detached mode
```

_Link to Docker Hub image:_ [farrandi/522-workflows-group-18](https://hub.docker.com/r/farrandi/522-workflows-group-18)

2. Click the link in the terminal to launch Jupyter Notebook. It should look something like this: `http://127.0.0.1:8888/lab`

3. Navigate to the analysis notebook at `work/notebooks/english_language_learning_ability_prediction_analysis.ipynb`

_Note_: when you want to close the container, press `ctrl + c` or `cmd + c` in the terminal and run the following command:

```bash
docker compose down
```

### 🖥️ Viewing the report

1. Make sure you have done the steps in the previous section and are in the jupyter lab UI (either via the `jupyter lab` command from the conda environment or via the link in the terminal when you run `docker compose up`)

2. Navigate to `notebooks/english_language_learning_ability_prediction_analysis.ipynb` to view our data analysis process, model training, and predictions.

3. Feel free to click `Restart Kernel and Run All Cells` to re-run the analysis.

### 🏃 Running the analysis

1. Make sure you followed Method 1 in the **Setting up your environment** section.

   1. If you are using **Method 1: conda environment**, make sure you see`(522)` at the beginning of the terminal line, like this: `(522) $username@computername:~$`
      <br />
   2. If you are using **Method 2: Docker container**, make sure you run the commands:

   ```bash
   docker compose run --rm analysis-nb-server bash
   ```

   Confirm this by checking that your terminal looks like: `jovyan@<some hash>:~/$`. To exit run `exit`.

<br />

2. Navigate to the project root and run the following command. (For Docker users, you should already be in the project root)

These are the simple commands with all the default values to run the analysis:

```bash
# clean up all the results from previous runs
make clean

# re-run the analysis
make all
```

#### 📝 Note

You can ignore steps 1 and 2 of the **Running the analysis** if you are using Docker. You can run the following command to run the analysis:

```bash
# clean up all the results from previous runs
docker compose run --rm analysis-nb-server make clean

# re-run the analysis
docker compose run --rm analysis-nb-server make all
```

---

### ✅ Testing the code

1. Make sure you followed Method 1 in the **Setting up your environment** section.

2. Navigate to the project root and run the following command.

```bash
python -m unittest discover tests
```

_Note_: There will be some windows that pop up when running the tests (You will need to close them to continue). This is expected behaviour.

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

- [Atabak Alishiri](https://github.com/atabak-alishiri)
- [Rachel Bouwer](https://github.com/rbouwer)
- [Salva Umar](https://github.com/salva-u)
- [Farrandi Hernando (Nando)](https://github.com/farrandi)

---

## 🤝 Contributing

Your contributions can help enhance this project further. For contribution guidelines, please refer to `CONTRIBUTING.md`. We appreciate your interest in improving the predictive capabilities of our model.

## ©️ License

All reports contained here are licensed under the [Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License](https://creativecommons.org/licenses/by-sa/4.0/). See [the license file](LICENSE.md) for more information.

If re-using/re-mixing please provide attribution and link to this webpage.

The software code contained within this repository is licensed under the MIT license. See [the license file](LICENSE.md) for more information.

The dataset employed in this analysis is distributed under an open-source license.

## 🙏 Acknowledgments

Special thanks to our course instructors and UBC for providing the resources and support necessary for this project.

## ✉️ Contact

For queries or collaborations, feel free to contact any of our team members.

You can find our contact details in the [Team Members](#-team-members) section.
