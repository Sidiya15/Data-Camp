# Data-Camp
Kaggle Competition - Ashrae-energy-prediction dataset

It's no longer secret that the climate change is a major global concern and a significant challange. Being aware of this reality and in order to meet this challenge, significant investments are being made today to improve the efficiency of buildings in order to reduce not only greenhouse gas emissions but also the cost. However, there is always this question regarding the effectiveness of these efforts.
It was in this context that a competition was launched by Kaggle and which consisted of developping accurate models of metered building energy usage in the following areas: chilled water, electric, hot water, and steam meters. The data comes from over 1,000 buildings over a three-year timeframe. With better estimates of these energy-saving investments, large scale investors and financial institutions will be more inclined to invest in this area to enable progress in building efficiencies.

We participated in this competition and the first thing we started with was an exploratory analysis in order to get more familiar with the data, which will allow us later to adopt the right strategies for managing missing values and feature engineering.
We tried different set of regression algorithms mainly from the scikit-learn library with the exception of Light Gradient Boosted Machine algorithm (LightGBM).
Having all of these algorithms at our disposal and after an appropriate feature engineering strategy, the goal was to just adjust their hyperparameters for better performance, just like someone could turn the knobs on an FM radio to get a clear signal. With that being said, it remains to specify the performance criteria to consider. In our case, we have chosen to consider the Mean Absulute Error (MAE), the Mean Absulute Percentage Error (MAPE) and Max Absulute Deviation (MAD) to evaluate the performance of all these models.

At the end, the model that was retained was the Random Forest algortihm with some parameters adjusted and the score obtained after the submission was 1.28
