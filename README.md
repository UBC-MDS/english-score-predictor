# 🌐 English Language Learning Ability Prediction

---

## 📔 About

Welcome to our project, "English Language Learning Ability Prediction", an innovative venture by a team of dedicated data scientists from the Master of Data Science program at the University of British Columbia. The goal of our project was to forecast an individual's aptitude for learning by predicting their performance in an English speaking test, which is part of a quiz to assess English proficiency. This prediction is based on demographic details and their linguistic background, including factors such as the duration of their stay in English-speaking countries, their primary and native languages, among others. We aimed to develop a regression model utilizing both Ridge and Lasso models. The effectiveness of these models was evaluated using metrics like R-squared, Root Mean Squared Error (RMSE), and Negative Mean Squared Error (NMSE).
We utilized a subset of a dataset originally compiled from 680,333 participants, ranging in age from 7 to 89, for our study. These individuals completed an English grammar quiz and provided demographic information, as well as details about their language backgrounds and the countries they have lived in. This dataset was central to the research paper "A critical period for second language acquisition: Evidence from 2/3 million English speakers" by Joshua K. Hartshorne, Joshua B. Tenenbaum, and Steven Pinker. The study focused on understanding the peak age for grammar-learning ability, observing its maintenance during childhood, and its notable decline in late adolescence. The authors performed various analyses and compared multiple models to identify age-related patterns in language acquisition. The complete dataset is accessible at [https://osf.io/pyb8s/](https://osf.io/pyb8s/). For the initial phase of our project (Milestone 1), we only used 30% of the total dataset. The script for sampling this subset is available in 'src/Sampling from dataset.ipynb'. In future project stages, we plan to consider incorporating the entire dataset. This rich dataset allows for a multi-faceted analysis of language learning patterns.

---

## 👥 Team Members

- Salva
- Atabak
- Nando
- Rachel

---

## 💻 Getting Started

### Dependencies

- Ensure you have `conda` installed as we use a conda environment to manage our project dependencies.
- Python and packages listed in [environment.yml]

### Installing

- Clone the repository:

```bash
git clone [https://github.com/UBC-MDS/522-workflows-group-18]
cd 522-workflows-group
```

- Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate 522
```

---

## 🚀 Usage

To dive into our analysis:

1. Launch Jupyter Notebook:

```bash
jupyter notebook English Language Learning Ability Prediction.ipynb
```

2. Navigate through the notebook to view our data analysis process, model training, and predictions.

---

## 🔍 Methodology

We employ a comprehensive approach:

1. **Data Preprocessing:** Cleaning, normalization, and transformation of data to ensure quality and consistency.
2. **Exploratory Data Analysis:** Utilizing statistical techniques and visualization tools to uncover trends and patterns.
3. **Model Development:** TO BE ADDED
4. **Model Evaluation:** TO BE ADDED

---

## 📈 Results and Discussion

TO BE ADDED

---

## 🤝 Contributing

Your contributions can help enhance this project further. For contribution guidelines, please refer to `CONTRIBUTING.md`. We appreciate your interest in improving the predictive capabilities of our model.

## ©️ License

This project is released under the MIT License, promoting open-source collaboration. For more details, see `LICENSE.md`.

## 🙏 Acknowledgments

Special thanks to our course instructors and UBC for providing the resources and support necessary for this project.

## ✉️ Contact

For queries or collaborations, feel free to contact any of our team members.
