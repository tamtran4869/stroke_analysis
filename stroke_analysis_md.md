---
title: "Stroke_analysis"
author: "Tam Tran"
date: '2023-01-31'
output: github_document
---

# Stroke analysis

## Import libraries

```{r setup, include= False}
install.packages('reticulate')
Sys.setenv(RETICULATE_PYTHON = "/usr/bin/python3")
library(reticulate)
```

```{r setup, include= False}
py_config()
use_python("/usr/bin/python3")

```

```{python}
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from shaphypetune import BoostRFE
import xgboost as xgb
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
```

## Datasets

```{python}
df = pd.read_csv("data.csv")
df.head()
```

The dataset includes 5110 records of 5110 cases, 1 `id`, 10 demographic and health indices, and 1 column indicating whether or not the patient had a stroke. The column `id` does not play a role in predicting stroke, so it should be removed.

```{python}
df.info()
```

There are 3 numerical factors: `age`,`avg_glucose_level` and `bmi`.

The others are categorical. However, these variables' types need to be adjusted. Some are noted as "int64," while others are "object" type. This problem is solved by converting to category type.

```{python}
df["stroke"].value_counts()
```

It is an imbalanced dataset which contains fewer positive cases.

## Pre-processing data

In this step, the dataset is cleaned, checked NAN, checked duplicates and inspected further inside each feature.

### Drop redundant features

```{python}
df = df.drop ("id",axis =1)
df.head()
```

The `id` column is dropped.

### Check NA values

```{python}
df.isna().sum()
print(' ')
print('%s NA values in positive cases' % df[df['bmi'].isna() & df['stroke'] == 1].shape[0])
```

There are 201 NA values in the `bmi` column, including 40 positive cases without `bmi` index. The NA problem is solved by filling with the mean value rather than deleting it due to the data imbalance with fewer positive cases.

```{python}
df['bmi'].fillna(np.round(df['bmi'].mean(), 1), inplace = True)
```

### Check duplicates

```{python}
dupli_sum = df.duplicated().sum()
if (dupli_sum>0):
    df = df.loc[False==df.duplicated(), :]
    print("Removed %s duplicates" % dupli_sum)
else:
    print("No duplicates found")

```

No duplicates are found in the dataset.

### Inspect categorical variables

```{python}
for i in [
    "gender",
    "work_type",
    "ever_married",
    "residence_type",
    "smoking_status",
    'stroke',
    'hypertension',
    'heart_disease'
]:
    print("---- %s ----" % i)
    df[i].value_counts()

    
```

In gender, the other category just has one record, which is not significant for the analysis, so it is eliminated. Moreover, as mentioned above, all categorical variables have "object" type which need to change into a "category" type as well.

```{python}
# Drop "Other" rows in gender
df = df[df["gender"] != "Other"]

# Change columns type
for i in [
    "gender",
    "work_type",
    "ever_married",
    "residence_type",
    "smoking_status",
    'stroke',
    'hypertension',
    'heart_disease'
]:
    print("-- %s --" % i)
    df[i] = df[i].astype('category')
```

### Inspect numerical variables

```{python}
for i in ["age","avg_glucose_level","bmi"]:
    print("-- %s --" % i)
    df[i].describe()
```

These variables have different scales.

Grouping into bins for visualisation purposes.

```{python}
df['age_binned'] = pd.cut(df['age'], np.arange(0, 91, 5))
df['avg_glucose_level_binned'] = pd.cut(df['avg_glucose_level'], np.arange(0, 301, 10))
df['bmi_binned'] = pd.cut(df['bmi'], np.arange(0, 101, 5))

#Final table
df.head()
```

## EDA

### Normalise and encode variables

For continuous variables, using min-max normalisation to convert all these data into the same scales [0,1].

```{python}
for i in ["age","avg_glucose_level","bmi"]:
    print("-- %s --" % i)
    df[i+"_norm"] = (df[i]-df[i].min())/(df[i].max()-df[i].min())
    df[i+"_norm"].describe()

df.head()  
```

For categorical variables, use a dummy encoder to convert the category into dummies and drop one of the dummies to avoid the dummy trap (perfect multicollinearity).

```{python}
dummies = pd.get_dummies(df[["gender", "work_type", "ever_married", "residence_type","smoking_status","hypertension","heart_disease"]])
                      
dummies = dummies.drop([
    "gender_Male",
    "work_type_children",
    'ever_married_No',
    'residence_type_Urban',
    "smoking_status_Unknown",
    "hypertension_0",
    "heart_disease_0"
], axis=1)

dummies.head()
```

Concat normalised columns and dummies into X (independent variables) and y (dependent variable )

```{python}
X = pd.concat([df[["age_norm","avg_glucose_level_norm","bmi_norm"]],dummies],axis=1)
y = df["stroke"]
print ('---- Independent factors------')
X.head()
print ('---- Dependent factors------')
y
```

### Correlation among X

Plotting a heatmap to show correlations between independent features.

```{python}
cor_matrix = round(X.corr(), 3)
sns.heatmap(
    cor_matrix,
    annot=True,
    cmap="magma_r"
)
plt.title("Correlation between independent variables", fontweight='bold')
plt.show()
```

There are many insights derived from the heatmap by noticing high positive (for positive relations) and high negative numbers (for inverse relations) between different variables.

-   Most inverse correlations come from dummies of variables due to trade-offs.

-   Ages have some noticed correlations. One obvious relation is between `age` & `ever_married_Yes`. Married persons are adults. Old people tend to have their own businesses and quit smoking.

-   3 numerical variables have positive relations. `bmi` and `avg_glucose_level` increase when people get old. There is a positive relation between `avg_glucose_level` and `bmi` but not strong as the relations with age.

-   Married persons often have high `bmi` and glucose levels. They have jobs and have been working.

-   `gender` and `residence_type` do not show any significance related to variables.

The Pearson correlation works for pairs of variables, so for further checking correlation, the VIF metric is adopted to understand the correlation of a variable with multiple others.

```{python}
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
## calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print(vif_data)
```

It shows the same pattern with the correlation heatmap.

### Explore the relations between X and y

Four functions are defined to generate a frequency chart, a percentage chart, a boxplot (for numeric only) and a function to combine 3 charts for convenience.

```{python}
def get_bar_chart(column, width=1, stacked=False):
    # Get the count of records by column and stroke
    df_pct = df.groupby([column, 'stroke'])['age'].count()
    # Create proper DataFrame's format
    df_pct = df_pct.unstack()
    return df_pct.plot.bar(
        stacked=stacked,
        figsize=(6, 6),
        width=width,
        ax=ax1,
        legend=False
    )

def get_100_percent_stacked_bar_chart(column, width=0.5):
    # Get the count of records by column and stroke
    df_breakdown = df.groupby([column, 'stroke'])['age'].count()
    # Get the count of records by gender
    df_total = df.groupby([column])['age'].count()
    # Get the percentage for 100% stacked bar chart
    df_pct = df_breakdown / df_total * 100
    # Create proper DataFrame's format
    df_pct = df_pct.unstack()
    return df_pct.plot.bar(
        stacked=True,
        figsize=(6, 6),
        width=width,
        ax=ax2,
        legend=False
    )

def get_boxplot_chart(column):
    # Get boxplot chart for continuos variables
    return sns.boxplot(
        data=df,
        x='stroke',
        y=column,
        ax=ax3
    )

def get_charts(column, type="numerical"):
    # Check the variables type
    if type not in ["numerical", "categorical"]:
        print("Please choose variable type numerical or categorical")
        return
    # Declare plot variables
    global fig, ax1, ax2, ax3
    # Plot numerical variable
    if type == "numerical":
        ori = "_".join(column.split("_")[:-1])
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
        ax1 = get_bar_chart(column, width=0.9)
        ax2 = get_100_percent_stacked_bar_chart(column, width=0.9)
        ax3 = get_boxplot_chart(ori)
        ax3.set_ylabel(ori)
        ax1.xaxis.set_tick_params(labelsize=9)
        ax2.xaxis.set_tick_params(labelsize=9)
    # Plot category
    if type == "categorical":
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1 = get_bar_chart(column, width=0.9)
        ax2 = get_100_percent_stacked_bar_chart(column, width=0.9)
        ax1.xaxis.set_tick_params(labelsize=9, labelrotation=0.5)
        ax2.xaxis.set_tick_params(labelsize=9, labelrotation=0.5)
    # Setup same formating parameter
    ax1.set_ylabel("frequency")
    ax2.set_ylabel("percentage")
    fig.legend(
        ["no stroke", "stroke"],
        loc="center right"
    )
```

Getting visualations of continuous values.

```{python,fig.height=4, fig.width=14}
get_charts("age_binned")
plt.suptitle("Visualisations for age and stroke")
plt.show()
```

Age strongly affects stroke conditions when seniors have higher stroke cases and stroke ratio. The stroke trend starts to increase dramatically after 35 years old. Most of the people suffering from stroke are greater than 70. There are some outliers, but it is not many.

Although `age` is highly correlated with other independent features, it seems to be potentially significant in predicting `stroke`.

```{python,fig.height=4, fig.width=14}
get_charts("avg_glucose_level_binned")
plt.suptitle("Visualisations for average glucose level and stroke")
plt.show()
```

The first chart shows the distribution by average glucose level, with many observations in the range of [60,110]. The second plot indicates the stroke ratio increases when the glucose level increases. Again in the third chart, stroke cases have a higher variance, with most cases having sugar levels greater than 100.

```{python,fig.height=4, fig.width=14}
get_charts("bmi_binned")
plt.suptitle("Visualisations for bmi and stroke")
plt.show()
```

There are not many cases that have significantly high `bmi`; hence the percentage chart gets some missing values. Most cases are in the range [15,40]. In contrast with the sugar level, the variance of stroke cases is less than the normal cases, focusing on groups 25-30.

Getting visualisation of categorical variables

```{python,fig.height=5, fig.width=10}
get_charts("gender", type="categorical")
plt.suptitle("Visualisations for gender and stroke")
plt.show()
```

There is no significant difference between the 2 genders in terms of stroke conditions.

```{python,fig.height=5, fig.width=10}
get_charts("work_type", type="categorical")
plt.suptitle("Visualisations for work type and stroke")
plt.show()
```

The dataset has the most stroked cases in private work type but the highest stroke ratio belongs to business owner groups. People working for governments are in the thread of stroke as well.

```{python,fig.height=5, fig.width=10}
get_charts("ever_married", type="categorical")
plt.suptitle("Visualisations for ever married and stroke")
plt.show()
```

Obviously, married people have a high stroke percentage which may be explained by the stress of marriage and the relation with age features (mentioned in the heatmap). Therefore, we need to run regression models to define significance for more accuracy.

```{python,fig.height=5, fig.width=10}
get_charts("residence_type", type="categorical")
plt.suptitle("Visualisations for residence type and stroke")
plt.show()
```

The places seem to be not much related to stroke.

```{python,fig.height=5, fig.width=10}
get_charts("smoking_status", type="categorical")
plt.suptitle("Visualisations for smoking status and stroke")
plt.show()
```

People who quit smoking usually suffer a high thread of stroke. It may be explained by the old persons tends to quit smoking and the effect of age, which makes the stroke ratio goes high (mentioned in the heatmap).

```{python,fig.height=5, fig.width=10}
get_charts("hypertension", type="categorical")
plt.suptitle("Visualisations for hypertension and stroke")
plt.show()
```

The distribution is an imbalance with more positive cases in a group of people with no hypertension but in the second chart, stroke appears in persons with hypertension more often than without it.

```{python,fig.height=5, fig.width=10}
get_charts("heart_disease", type="categorical")
plt.suptitle("Visualisations for heart disease and stroke")
plt.show()
```

It shows the same pattern with the feature `hypertension` but has higher level differences in the 2 groups of stroke.

## Modeling

### Balance the dataset

To balance data, SMOTE model generates synthetic data based on the distribution of training and testing set

```{python}
# Split dataset
train_x_full, test_x_full, train_y, test_y = train_test_split(
    X,
    y,
    random_state=1255,
    test_size=0.25
)

# Get SMOTE model and generate synthetic data
smote = SMOTE()
train_x_full, train_y = smote.fit_resample(train_x_full, train_y)
test_x_full, test_y = smote.fit_resample(test_x_full, test_y)

# Print out size of new datasets
print(
    train_x_full.shape,
    train_y.shape,
    test_x_full.shape,
    test_y.shape
)
```

###Fit model

for testing and evaluating different models eaiser, 2 functions are defined: 1 for plot ROC curves and 1 for prediction and evaluation.

```{python}
# ROC ploting function
def plot_roc_curve(true_y, y_prob, auc, type):
    # Plot the roc curve based of the probabilities
    fpr, tpr, _ = roc_curve(true_y, y_prob[:, 1])
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(label="Model %s (AUC: %s)" % (type, auc))
    return None
```

```{python}
# Test and evaluation model function
def test_model(m_dict):
    # Declare variables
    type = m_dict["type"]
    model = m_dict["model"]
    y_train = train_y
    y_test = test_y
    x_train = train_x_full.drop(m_dict["removed_features"], axis=1)
    x_test = test_x_full.drop(m_dict["removed_features"], axis=1)
    # Train model and get predictions
    clf = model.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)
    # Compute evaluation metrics
    scores = cross_val_score(clf, x_train, y_train, cv=10)
    accuracy_mean = scores.mean()
    cfm = pd.crosstab(
        y_test,
        y_pred,
        rownames=['Actual'],
        colnames=['Predicted'])
    auc = round(roc_auc_score(y_test, y_pred), 3)
    # Print out results
    print("-----------Model: %s ------------" % type)
    print(" ")
    print("K-Fold Validation Mean Accuracy: %s " % accuracy_mean)
    print(" ")
    print("Confusion matrix: \n %s " % cfm)
    print(" ")
    print("Evaluation metrics: \n %s " % classification_report(y_test, y_pred))
    # Get ROC plot for model
    roc = plot_roc_curve(y_test, y_proba, auc, type)
    plt.show()
    return y_pred, accuracy_mean, cfm, auc, roc

```

The testing function takes model types, model initiation and desired removed features (if not, use the empty list []) as an argument dictionary to return predictions, accuracy mean of cross-validation with k = 10, confusion matrix, AUC values and ROC curves.

#### Logistic Regression

Due to the dependent variable being binary variables, logistic regression is suitable, but it needs to check with the assumptions of the regression model. As explored above, there are some highly correlated features needed to be addressed, so this stage includes assumption checking and feature selection by using the `GLM()` function.

Basic assumptions of logistic regression: linearity in the logit for continuous variables, independence of errors, absence of multicollinearity, and lack of strongly influential outliers. (Note: The multicollinearity mentioned in the correlation analysis above.)

```{python}
res = GLM(
    train_y,
    train_x_full,
    family=families.Binomial(),
).fit(random_state=0)
```

The linearity assumption is checked by visualisation of continuous features with the log odds.

```{python, fig.height=6, fig.width=12}
# ASSUMPTION: Linearity logit relation
pred_x= res.predict(train_x_full)
# Getting log odds values
log_odds = np.log(pred_x / (1 - pred_x))
# Visualize predictor variable vs logit values for Age
fig, (ax1,ax2,ax3) = plt.subplots(nrows = 3)
ax1.scatter(x=train_x_full['age_norm'].values, y=log_odds)
ax2.scatter(x=train_x_full['bmi_norm'].values, y=log_odds)
ax3.scatter(x=train_x_full['avg_glucose_level_norm'].values, y=log_odds)
ax1.set_xlabel("Age")
ax1.set_ylabel("Log-odds")
ax2.set_xlabel("BMI")
ax2.set_ylabel("Log-odds")
ax3.set_xlabel("Average Glucose Level")
ax3.set_ylabel("Log-odds")
plt.show()
```

It is clear that all continuous variables have a linear relation with the log odds.

There are many indexes for checking influential points, and in this case, we use Cook's Distance (a measure of the simultaneous change in the parameter estimates when an observation is deleted from the analysis) and Standardised Residuals (residuals divided by their standard errors). Computing these metrics for each observation and checking whether they exceed the threshold. All observations that exceed the threshold are influential points.

```{python}
# ASSUMPTION:Influential points
infl = res.get_influence(observed=False)
summ_df = infl.summary_frame()

cook_threshold = 4 / len(train_x_full)
summ_df['standard_resid'] = summ_df['standard_resid'].apply(lambda x: np.abs(x))
summ_df[(summ_df['cooks_d'] > cook_threshold) & (summ_df['standard_resid']>3)]

''' For  visualisation
fig = plt.subplots()
infl.plot_index(y_var='cooks', threshold=cook_threshold)
infl.plot_index(y_var='resid', threshold=3)
fig.tight_layout(pad=1.0)
plt.show()
'''
```

Only 118 cases are outside the thresholds of Cook's Distance and standardised residuals. It is less than 5% of the number of cases (5%\*7294 = 365), so the influential points may not cause any problems with the model. The assumption is satisfied.

The independence of error terms is validated by checking the chart of residuals and their order. If there are any special patterns that means observations, errors are linked to each other

```{python}
# ASSUMPTION : Independence of errors
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, title="Residual Series Plot",
                     xlabel="Index Number", 
                     ylabel="Deviance Residuals")
# Generate residual series plot using standardized deviance residuals
ax.plot(train_x_full.index.tolist(), stats.zscore(res.resid_deviance))
# Draw horizontal line at y=0
plt.axhline(y = 0, ls="--", color='red')
plt.show()
```

The plot shows random errors. After nearly the first 4000 observations, the residuals increased, and it is explained by the synthetic data giving higher errors. However, generally, it still is scattered randomly.

Next is the feature selection.

```{python}
res = GLM(
    train_y,
    train_x_full,
    family=families.Binomial(),
).fit(random_state=0)

print(res.summary())
```

`work_type_Never_worked`, `ever_married_Yes` and `hypertension_1` are not significant in stroke predictions.

```{python}
removed_features = ['work_type_Never_worked', 'ever_married_Yes','hypertension_1']

res = GLM(
    train_y,
    train_x_full.drop(removed_features, axis=1),
    family=families.Binomial(),
).fit(random_state=0)
print(res.summary())
```

All variables are significant now. Although the good fit of the model is slightly decreased model only runs 6 iterations to converge. It helps save sources (time, memory) and avoid overfitting.

`LogisticRegression()` is a model initiation and is stored inside a dictionary with the type and list of removed features.

```{python}

logit_dict = {
    "type": "Logistic Regression",
    "model": logreg,
    "removed_features": removed_features
}
```

Using the testing function to get prediction and all evaluation metrics.

```{python3}
y_pred, accuracy_mean, cfm, auc, roc = test_model(logit_dict)
```

### XGBoost

XGBoost stands for Extreme Gradient Boosting. It is an alrothism in boosting essemble method where fit models iteratively and update the current model with the opposite of the gradient of fitting error (e.g. pseudo-residuals).

Creating an XGBoost model, using feature selections with `BoostRFE`, storing all required data inside a dictionary and testing with the function

```{python}
# Create model & select features.
xg = xgb.XGBClassifier(random_state=0)
xg_rfe = BoostRFE(xg)
xg_rfe.fit(train_x_full, train_y)
removed_features = [k for k, v in zip(train_x_full.columns, xg_rfe.ranking_) if v != 1]

# Store info
xg_dict = {
    "type": "XGBoost",
    "model": xg,
    "removed_features": removed_features
}

# Test model 
y_pred, accuracy_mean, cfm, auc, roc = test_model(xg_dict)
```

The XGBoost model (accuracy ~ 94%) performs better than the logistic regression (accuracy ~ 80%) in this case.

# References: 
<https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290#293b>

<https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8>

<https://www.analyticsvidhya.com/blog/2021/05/how-to-create-a-stroke-prediction-model/>

