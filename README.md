# House_Price_Prediction


![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white) 
![Pandas](https://img.shields.io/badge/Pandas-1.5-brightgreen) 
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-purple)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.2-orange)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)


A **machine learning project** that predicts house prices based on various property attributes. Includes **data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and visualization of results**.  

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#HousingDataset)  
- [Exploratory Data Analysis](#exploratory-data-analysis)   
- [Tools & Libraries](#Tools&Libraries)
- [Streamlit App](#StreamlitApp)
- [How to Run](#how-to-run)  
- [Conclusion](#conclusion)  
- [Author](#author)  
## Project Overview

This project provides an in-depth analysis of the housing market by building robust predictive models to accurately forecast property valuations. The process begins with rigorous data preprocessing, where we clean the dataset, manage missing values, and engineer new features to enhance model performance. Through comprehensive exploratory data analysis (EDA) and dynamic visualizations, we identify key trends, correlations, and underlying patterns. We then implement and evaluate various machine learning algorithms, from linear regression to gradient boosting, to determine the most effective model. The final outcome is a powerful tool that not only estimates house prices but also reveals the most influential factors that drive market values.


## 🏠 Housing Dataset 

**Dataset Source:** [Housing Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)

- price: The sale price of the house in Indian Rupees (target variable).
- area: The total built-up area of the house in square feet.
- bedrooms: The number of bedrooms in the house.
- bathrooms: The number of bathrooms in the house.
- stories: The number of floors (stories) in the house.
- mainroad: Whether the house is located on the main road (yes / no).
- guestroom: Whether the house has a separate guest room (yes / no).
- basement: Whether the house has a basement (yes / no).
- hotwaterheating: Availability of hot water heating facility (yes / no).
- airconditioning: Whether the house has an air conditioning system (yes / no).
- parking: The number of parking spaces available with the house (0–3).
- prefarea: Whether the house is located in a preferred area (yes / no).
- furnishingstatus: The furnishing condition of the house (furnished, semi-furnished, unfurnished).

## Exploratory Data Analysis

1. **Distribution of House Prices**  
   - Plotted histogram and KDE curve of prices.  
   - Checked for skewness and outliers.  

2. **Correlation Heatmap of Numerical Variables**  
   - Generated correlation matrix for numerical features (`price, area, bedrooms, bathrooms, stories, parking`).  
   - Observed strong correlations (e.g., `area` positively correlated with `price`).  

3. **Count Plots for Categorical Variables**  
   - Visualized category distribution for features like `mainroad`, `guestroom`, `basement`, `prefarea`, etc.  
   - Identified imbalances (e.g., very few houses with hotwaterheating).  

4. **Boxplots to Compare Price with Categorical Features**  
   - Compared `price` distributions across categorical variables.  
   - Found that **preferred area, mainroad, airconditioning, and furnishingstatus** significantly impact house prices.  

5. **Feature Importance Analysis**  
   - Encoded categorical variables for modeling.  
   - Applied **Random Forest Regressor** to rank features.  
   - Found **area, bathrooms, and furnishingstatus** to be strong predictors of house price. 

## Tools & Libraries
- **Python**
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Data visualization
- **Jupyter Notebook** – Interactive coding environment
- **Streamlit** - For Interactive Dashboard
 

## 🌐 Streamlit App  

I have also built an interactive **Streamlit web application** for this project.  
The app allows users to:  
- Explore the dataset visually.  
- View distribution plots, heatmaps, and boxplots.  
- Compare house prices based on categorical features.  
- Check feature importance rankings.  
- Predict house prices by entering custom inputs.  

🔗 **Live Demo:** [Streamlit App Link](https://housepriceprediction-16.streamlit.app/)  



## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Shrilaxmi-16/House_Price_Prediction.git

   pip install -r requirements.txt

   streamlit run app.py


## Conclusion

The analysis of the housing dataset revealed that:  

- **Area** of the house is the strongest factor influencing price.  
- Houses with **more bathrooms** and **multiple stories** tend to have higher prices.  
- Location-related features such as being on the **mainroad** or in a **preferred area** significantly increase the house value.  
- Amenities like **air conditioning** and **furnishing status** also contribute to higher prices.  
- Features such as **hot water heating** have little to no impact due to their rarity in the dataset.  

Overall, both **numerical factors (area, bathrooms, stories)** and **categorical features (location, furnishing, AC)** play a key role in determining housing prices. These insights can help buyers, sellers, and real estate professionals in making data-driven decisions.  

## Author
Shrilaxmi Gidd
Email: shrilaxmigidd16@gmail.com

Date: 25-09-2025
  
