from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
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

'''
echo "# healthcare_statistical_analysis" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/tamtran4869/healthcare_statistical_analysis.git
git push -u origin main
'''

df = pd.read_csv("data.csv")

# Check type, clean data and add columns
df.describe()
df.info()

#------PRE-PROCESSING_---------
# Drop columns
df = df.drop("id", axis=1)

# Check nan values in the dataframe and fill nan values by its' mean
df.isna().sum()
df[df['bmi'].isna() & df['stroke'] == 1].shape[0]
df['bmi'].fillna(np.round(df['bmi'].mean(), 1), inplace=True)

# Check duplicates
dupli_sum = df.duplicated().sum()
if(dupli_sum > 0):
    df = df.loc[False == df.duplicated(), :]
    print("Removed %s duplicates" % dupli_sum)
else:
    print("No duplicates found")

# Inspect categorical parameters and change type from object to category
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

df = df[df["gender"] != "Other"]

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

# Create dummies variables from categorical variables and drop some dummies to avoid dummy trap
dummies = pd.get_dummies(df[["gender", "work_type", "ever_married",
                         "residence_type", "smoking_status", "hypertension", "heart_disease"]])
dummies = dummies.drop([
    "gender_Male",
    "work_type_children",
    'ever_married_No',
    'residence_type_Urban',
    "smoking_status_Unknown",
    "hypertension_0",
    "heart_disease_0"
], axis=1)

# Add binned columns for numerical columns for visualisation
for i in ["age", "avg_glucose_level", "bmi"]:
    print("-- %s --" % i)
    df[i].describe()

df['age_binned'] = pd.cut(df['age'], np.arange(0, 91, 5))
df['avg_glucose_level_binned'] = pd.cut(
    df['avg_glucose_level'], np.arange(0, 301, 10))
df['bmi_binned'] = pd.cut(df['bmi'], np.arange(0, 101, 5))

# Get normalised numerical columns and statistics
for i in ["age", "avg_glucose_level", "bmi"]:
    print("-- %s --" % i)
    df[i+"_norm"] = (df[i]-df[i].min())/(df[i].max()-df[i].min())
    df[i+"_norm"].describe()


# Concat with numerical variable and split dataframe into X (independent variables) and y(dependent variables)
X = pd.concat(
    [df[["age_norm", "avg_glucose_level_norm", "bmi_norm"]], dummies], axis=1)
y = df["stroke"]

# ----EDA------

# Investigate correlation between independent variables.
# Get correlation heatmap
cor_matrix = round(X.corr(), 3)
sns.heatmap(
    cor_matrix,
    annot=True,
    cmap="magma_r"
)
plt.title("Correlation between independent variables", fontweight='bold')
plt.show()

# sns.pairplot(df[["age_norm","avg_glucose_level_norm","bmi_norm"]])

# Check VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_data)

# Relation with the dependent variable y - stroke

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
        ncol=2,
        loc="lower center"
    )

# Numerical variables
get_charts("age_binned")
plt.suptitle("Visualisations for age and stroke")
plt.show()

get_charts("avg_glucose_level_binned")
plt.suptitle("Visualisations for average glucose level and stroke")
plt.show()

get_charts("bmi_binned")
plt.suptitle("Visualisations for bmi and stroke")
plt.show()


# Categorical variables

get_charts("gender", type="categorical")
plt.suptitle("Visualisations for gender and stroke")
plt.show()

get_charts("work_type", type="categorical")
plt.suptitle("Visualisations for work type and stroke")
plt.show()

get_charts("ever_married", type="categorical")
plt.suptitle("Visualisations for ever married and stroke")
plt.show()

get_charts("residence_type", type="categorical")
plt.suptitle("Visualisations for residence type and stroke")
plt.show()

get_charts("smoking_status", type="categorical")
plt.suptitle("Visualisations for smoking status and stroke")
plt.show()

get_charts("hypertension", type="categorical")
plt.suptitle("Visualisations for hypertension and stroke")
plt.show()

get_charts("heart_disease", type="categorical")
plt.suptitle("Visualisations for heart disease and stroke")
plt.show()

# ----BALANCE DATASET-----
train_x_full, test_x_full, train_y, test_y = train_test_split(
    X,
    y,
    random_state=1255,
    test_size=0.25
)

smote = SMOTE()
train_x_full, train_y = smote.fit_resample(train_x_full, train_y)
test_x_full, test_y = smote.fit_resample(test_x_full, test_y)

print(
    train_x_full.shape,
    train_y.shape,
    test_x_full.shape,
    test_y.shape
)

# -----CREATE MODEL ----------
def plot_roc_curve(true_y, y_prob, auc, type):
    # plot the roc curve based of the probabilities
    fpr, tpr, _ = roc_curve(true_y, y_prob[:, 1])
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'r--')
    #plt.grid()
    plt.title(label="Model %s (AUC: %s)" % (type, auc))


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
        colnames=['Predicted']
        )
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


# Logistic regression
res = GLM(
    train_y,
    train_x_full,
    family=families.Binomial(),
).fit(random_state=0)
print(res.summary())

# Assumption: Linearity logit relation
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

# Assumption: Multicollinearity mentioned above in the correlation part

# Assumption:Influential points
infl = res.get_influence(observed=False)
summ_df = infl.summary_frame()
summ_df

cook_threshold = 4 / len(train_x_full)
summ_df['standard_resid'] = summ_df['standard_resid'].apply(lambda x: np.abs(x))
summ_df[(summ_df['cooks_d'] > cook_threshold) & (summ_df['standard_resid']>3)]

summ_df.sort_values("cooks_d", ascending=False)[:10]
fig = plt.subplots()
infl.plot_index(y_var='cooks', threshold=cook_threshold)
infl.plot_index(y_var='resid', threshold=3)
fig.tight_layout(pad=1.0)
plt.show()

# Assumption: Independence of observations
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, title="Residual Series Plot",
                     xlabel="Index Number", 
                     ylabel="Deviance Residuals")
# Generate residual series plot using standardized deviance residuals
ax.plot(train_x_full.index.tolist(), stats.zscore(res.resid_deviance))
# Draw horizontal line at y=0
plt.axhline(y = 0, ls="--", color='red')
plt.show()

# Feature selection
removed_features = ['work_type_Never_worked', 'ever_married_Yes','hypertension_1']
train_x = train_x_full.drop(removed_features, axis=1)

res = GLM(
    train_y,
    train_x,
    family=families.Binomial(),
).fit(random_state=0)
print(res.summary())

# Traing and test model
logreg = LogisticRegression(random_state=0)

logit_dict = {
    "type": "Logistic Regression",
    "model": logreg,
    "removed_features": removed_features
}

y_pred, accuracy_mean, cfm, auc, roc = test_model(logit_dict)

# XGBoost
xg = xgb.XGBClassifier(random_state=0)
xg_rfe = BoostRFE(xg)
xg_rfe.fit(train_x_full, train_y)
removed_features = [k for k, v in zip(train_x_full.columns, xg_rfe.ranking_) if v != 1]

xg_dict = {
    "type": "XGBoost",
    "model": xg,
    "removed_features": removed_features
}
y_pred, accuracy_mean, cfm, auc, roc = test_model(xg_dict)

'''
References:
https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290#293b
https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
https://www.analyticsvidhya.com/blog/2021/05/how-to-create-a-stroke-prediction-model/

'''
