---
title: Home
layout: home
nav_order: 1
---

# Home Credit Risk Analysis
This project is a part of the CS 7641 - Machine Learning course at Georgia Tech. The goal of this project is to analyze home credit risk using machine learning techniques, and provide predictions on whether or not an individual will default on a home loan based on a multitude of factors, ranging from external source (credit bureaus) investigations to individual income to demographic factors (age, gender, location, etc.). This project was completed by by [Reetesh Sudhakar](https://www.github.com/reeteshsudhakar), [Nityam Bhachawat](https://github.com/nityamb), [Mark Glinberg](https://github.com/mng03), and [Yash Gupta](https://github.com/hashgupta). Iterations and documentation of the project can be discovered on this website, as well as a detailed analysis of the final iteration of our models and techniques employed to analyze the dataset. The code can be found on our [GitHub repository](https://github.com/reeteshsudhakar/CS-7641-Project). 

With housing markets facing uncertainty and limited credit histories hindering borrowers, this analysis helps ensure qualified applicants aren’t unfairly denied while identifying likely defaulters early. Manual risk evaluations are inconsistent and time-consuming, necessitating automated, unbiased assessments.  

To categorize individuals by repayment risk, the project utilized Home Credit’s extensive dataset of over 300,000 applicants with 100+ attributes like income, employment, credit history, and existing credits. After substantial data cleaning and preprocessing, multiple classification algorithms were trained, including Support Vector Machines, Logistic Regression, and Random Forests. Dimensionality reduction using Principal Component Analysis also enabled efficient handling of the high-dimensional data.

The classifiers demonstrated competent performance as measured by metrics like balanced accuracy, precision, recall, and F1-scores. Balanced accuracy, which handles class imbalance better than raw accuracy, ranged from 63-68%. High precision scores up to 0.895 reveal effective identification of defaulters with low false positive rates. Though some challenges remain in further improving true positive rates, these methods significantly outperform a naive baseline model. 

Overall, the analyses provide actionable insights for financial institutions to streamline evaluations and lending decisions. Automated risk scoring enhances consistency, reduces biases, and lowers defaults. With tailored feature engineering and model tuning, greater predictive capabilities could be attained. The frameworks developed serve as a robust foundation for scaled deployment across lending platforms. By reliably determining creditworthiness early on, this work has far-reaching implications in broadening access to housing finance while preventing exploitation.