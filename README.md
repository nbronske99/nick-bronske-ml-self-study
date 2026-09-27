# nick-bronske-ml-self-study
I am a math student in my first year at Bunker Hill Community College.
This repo will serve as a documented place where I will be self studying various ideas in machine learning.

## Contents
| Script | What it does |
|---|---|
| `linear_regression.py` | Fits a line to BMI vs. glucose by hand (least squares) and reports R². |
| `newtons_method.py` | Newton's method for root finding and optimization, compared with gradient descent. |
| `logistic_regression.py` | Predicts whether a person has diabetes, trained with Newton's method and tested on held-out people. |

Data: `diabetes.csv` is the Pima Indians Diabetes dataset (768 people, `Outcome` = 1 if diabetic).

## Running
```
pip install -r requirements.txt
python logistic_regression.py
```
Each script saves its plot as a `.png` in the repo folder.
