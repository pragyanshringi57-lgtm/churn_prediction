I am still working on this 
this is my first project and first github repo so code is quite simple and unorganized.Right now, the pipeline is running a baseline Linear Regression model with an accuracy of around 80%.  the next step will include multiple different model and inclusive code
so with some help of ai to organize it it's split into 3 part 
* **`notebooks/EDA.ipynb`**: My scratchpad. This is where I did the initial data exploration and built the charts (like tenure distribution and gender impact) to actually understand the dataset.
* **`src/preprocess.py`**: The cleanup crew. This script takes the raw CSV, handles missing values (like blanks in the charges column), fixes some categorical quirks, and one-hot encodes the text into numbers.
* **`src/train.py`**: The main event. It imports the clean data from the preprocess script, does the train/test split, and fits the model.

 Tech Stack
Just the standard data science toolkit: Python, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.

Right now, the pipeline is running a baseline Linear Regression model with an accuracy of around 80%. 
 My next steps are:
1. Swapping the baseline for a proper classifier like XGBoost.
2. Changing the evaluation metric from Accuracy to F1-Score/Recall.
3. Adding SMOTE to handle the class imbalance (since most people in the dataset didn't churn
