# Home Credit Risk Analysis

## Project Overview

This project aims to provide insights towards an individual's likelihood of defaulting on a home loan. Using a Kaggle dataset of 300,000+ observations, this project explores the data, cleans it, and builds different models to predict whether an individual will default. 

This project was completed as a part of a semester-long project for CS 7641 (Machine Learning) at Georgia Tech in the Fall 2023 semester by [Reetesh Sudhakar](https://www.github.com/reeteshsudhakar), [Nityam Bhachawat](https://github.com/nityamb), [Mark Glinberg](https://github.com/mng03), and [Yash Gupta](https://github.com/hashgupta). 

## Navigating the Repository
Many files in the repository are for the web page associated with this project, which can be accessed at the [following page](http://www.reeteshsudhakar.com/CS-7641-Project). Modifications can be made to the [_config.yml](/_config.yml) file to change the theme, title, and other aspects of the web page. 

Files in the resources folder are used to store images and other files that are used in the web page. 

You can access the code used to clean the data and build the models in the [code folder](/code). To run the code, create a **data folder** called "data" in the src directory (`src/data`) and download the dataset from [this link](https://www.kaggle.com/competitions/home-credit-default-risk/data) to ensure that you can clean the data and use it with our written code. This directory was added to our .gitignore file, as the code is set up such that users can download the data and run the code to clean it themselves without having to push the data to GitHub.

If you'd rather not do any of the data processing and choose to not modify it, you can simply read the data from the [cleaned data file](/src/cleaned_data.csv) and use it to build the models.

Helper functions are located in both [util.py](/src/util.py) and [get_feature_importance.py](/src/get_feature_importance.py). Code for each of the classifiers is located in the [classifier.py](/src/classifier.py) file. Visualizations of the dataset can be found in the [visualization notebook](/src/data_visualization.ipynb). 