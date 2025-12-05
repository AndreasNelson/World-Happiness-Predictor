# World Happiness Predictor

A machine learning-powered Shiny dashboard that predicts national happiness scores ("Life Ladder") based on economic, social, and health indicators.

**Live Demo:** [https://andreas-nelson.shinyapps.io/WorldHappiness/](https://andreas-nelson.shinyapps.io/WorldHappiness/)

## 📖 Overview

This project explores the drivers of global happiness by harmonizing data from the **World Happiness Report (Gallup World Poll)** and the **World Bank (WDI)**. Using a Random Forest regression model, the dashboard allows users to simulate how changes in national metrics—such as GDP per capita, social support, and corruption—impact a country's predicted happiness score.

## ✨ Key Features

* **Happiness Simulator:** Interactive sliders allow users to tweak national indicators to generate real-time happiness predictions.
* **Country Presets:** Pre-load historical data from specific countries to see how their actual metrics translate into predicted happiness.
* **Model Diagnostics:** Visualize model performance using Variable Importance Plots (VIP) and Predicted vs. Actual scatterplots.
* **Data Explorer:** A searchable, interactive table of the historical dataset (2005–2020).

## 🛠️ Tech Stack

* **Language:** R
* **Framework:** Shiny (UI/Server)
* **Machine Learning:** `tidymodels`, `ranger` (Random Forest), `recipes`
* **Data Manipulation:** `tidyverse`, `janitor`
* **Visualization:** `ggplot2`, `vip`

## 🧠 Methodology

### 1. Data Collection & Preprocessing
The dataset combines subjective well-being metrics with objective economic indicators.
* **WHR (Gallup):** Life Ladder, Social Support, Freedom to Make Life Choices, Generosity, Perceptions of Corruption.
* **World Bank:** Log GDP per Capita, Healthy Life Expectancy at Birth.
* *Merging:* An inner join was performed on `country_name` and `year` to align economic context with happiness surveys.

### 2. Modeling Strategy
* **Algorithm:** Random Forest Regression (`ranger` engine, 100 trees).
* **Feature Engineering:** Median imputation was used for missing predictors; near-zero variance predictors were removed.
* **Performance:** The model captures non-linear interactions between economic stability and social factors, with Log GDP and Healthy Life Expectancy being the strongest predictors.

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/world-happiness-predictor.git](https://github.com/yourusername/world-happiness-predictor.git)
    ```

2.  **Open `app.R` in RStudio.**

3.  **Install dependencies:**
    ```r
    install.packages(c("shiny", "shinythemes", "tidyverse", "rsample", "recipes", "parsnip", "workflows", "tune", "yardstick", "ranger", "vip", "DT"))
    ```

4.  **Run the App:**
    Click the "Run App" button in RStudio or run:
    ```r
    shiny::runApp()
    ```

## 📂 File Structure

* `app.R`: Contains the full UI, Server logic, and model training pipeline.
* `master_data.rds`: The pre-processed dataset used for training and visualization.
* `README.md`: Project documentation.
