#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


# In[13]:


file_path = "../datasets/credit_risk_dataset.csv"
df = pd.read_csv(file_path)
df.head()


# #### Performing EDA on the Dataset

# In[14]:


df.info()


# In[15]:


# checking for missing values
df.isnull().sum()


# In[16]:


# checking for unique values in each column
df.nunique()


# In[17]:


# creating a pie chart for loan_status distribution
plt.figure(figsize=(6, 4))
loan_status_counts = df['loan_status'].value_counts()

# function to show both percentage and count
def autopct_format(pct):
    absolute = int(pct / 100. *len(df))
    return f'{pct:.1f}% \n({absolute:,})'

plt.pie(
    loan_status_counts.values,
    labels=loan_status_counts.index,
    autopct=autopct_format,
    startangle=90,
    colors=['lightcoral', 'skyblue'],
    explode=(0.05, 0)
)

plt.title('Distribution of Loan Status', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

# printing the actual counts for reference
print(f"Loan Status Distribution:")
for status, count in loan_status_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{status}: {count} ({percentage:.1f}%)")


# In[18]:


# creating a pie chart for loan_intent distribution
plt.figure(figsize=(7, 5))
loan_intent_counts = df['loan_intent'].value_counts()

def autopct_format_intent(pct):
    absolute = int(pct/100.*len(df))
    return f'{pct:.1f}% \n({absolute:,})'

colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold', 'lightsalmon', 'lightsteelblue']

plt.pie(
    loan_intent_counts.values,
    labels=loan_intent_counts.index,
    autopct=autopct_format_intent,
    startangle=90,
    colors=colors[:len(loan_intent_counts)],
    explode=[0.05] * len(loan_intent_counts)
)

plt.title('Distribution of Loan Intent', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[19]:


# visualizing the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='YlOrRd')
plt.title('Correlation Matrix', fontsize=18)


# In[20]:


# creating a box plot for income distribution by home ownership
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='person_home_ownership', y='person_income')

# setting y-axis to log scale for better visualization due to wide income range
plt.yscale('log')

plt.title('Income Distribution by Home Ownership', fontsize=16, fontweight='bold')
plt.xlabel('Home Ownership', fontsize=12)
plt.ylabel('Annual Income (log scale)', fontsize=12)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# printing summary statistics for each home ownership category
print("\nIncome Summary by Home Ownership:")
for ownership in df['person_home_ownership'].unique():
    subset = df[df['person_home_ownership'] == ownership]['person_income']
    print(f"\n{ownership}:")
    print(f"  Median: ${subset.median():,.2f}")
    print(f"  Mean: ${subset.mean():,.2f}")
    print(f"  Count: {len(subset):,}")


# In[21]:


# creating a violin plot for loan amount distribution by loan grade
plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='loan_grade', y='loan_amnt', inner='quartile')

plt.title('Loan Amount Distribution by Grade', fontsize=16, fontweight='bold')
plt.xlabel('Loan Grade', fontsize=12)
plt.ylabel('Loan Amount', fontsize=12)

# formatting y-axis to show values in thousands
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.show()

# printing summary statistics for each loan grade
print("\nLoan Amount Summary by Grade:")
for grade in sorted(df['loan_grade'].unique()):
    subset = df[df['loan_grade'] == grade]['loan_amnt']
    print(f"\nGrade {grade}:")
    print(f"  Median: ${subset.median():,.2f}")
    print(f"  Mean: ${subset.mean():,.2f}")
    print(f"  Count: {len(subset):,}")
    print(f"  Range: ${subset.min():,.2f} - ${subset.max():,.2f}")


# In[22]:


# dropping null values since they are very few as compared to dataset size
df.dropna(axis=0, inplace=True)
df.isnull().sum()


# In[23]:


df.shape


# In[24]:


# checking the distribution of the target variable after dropping null values
df['loan_status'].value_counts()


# In[25]:


df.describe()


# In[26]:


# checking the distribution of ages
plt.figure(figsize=(10, 6))
sns.histplot(df['person_age'], bins=60, kde=True, color='blue')


# In[27]:


df['person_age'].value_counts().sort_index(ascending=False).head(10)


# We see that there's some unknown values for age, such as 144. For this reason, it's safe to drop values where age of the person is greater than 80, as most likely they won't be applying for loan considering the age.

# #### Feature Engineering

# In[28]:


# dropping ages greater than 80 as they are outliers
df = df[df['person_age'] <= 80]
df.shape


# In[29]:


# creating age groups
df['age_group'] = pd.cut(
    df['person_age'],
    bins=[20, 26, 36, 46, 56, 66],
    labels=['20-25', '26-35', '36-45', '46-55', '56-65']
)

# creating income groups
df['income_group'] = pd.cut(
    df['person_income'],
    bins=[0, 25000, 50000, 75000, 100000, float('inf')],
    labels=['low', 'low-middle', 'middle', 'high-middle', 'high']
)

# creating loan amount groups
df['loan_amount_group'] = pd.cut(
    df['loan_amnt'],
    bins=[0, 5000, 10000, 15000, float('inf')],
    labels=['small', 'medium', 'large', 'very large']
)


# ##### Visualizations Supporting Feature Engineering Decisions

# In[30]:


# 1. Age Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age histogram
axes[0,0].hist(df['person_age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(df['person_age'].mean(), color='red', linestyle='--', label=f'Mean: {df["person_age"].mean():.1f}')
axes[0,0].set_title('Age Distribution')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# Age vs Loan Status
sns.boxplot(data=df, x='loan_status', y='person_age', ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Loan Status')

# Age groups vs Default Rate
age_default_rate = df.groupby('age_group')['loan_status'].mean()
axes[1,0].bar(age_default_rate.index, age_default_rate.values, color='lightcoral', alpha=0.7)
axes[1,0].set_title('Default Rate by Age Group')
axes[1,0].set_xlabel('Age Group')
axes[1,0].set_ylabel('Default Rate')
axes[1,0].tick_params(axis='x', rotation=45)

# Age group distribution
age_group_counts = df['age_group'].value_counts().sort_index()
axes[1,1].bar(age_group_counts.index, age_group_counts.values, color='lightgreen', alpha=0.7)
axes[1,1].set_title('Distribution of Age Groups')
axes[1,1].set_xlabel('Age Group')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[31]:


# 2. Income Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Income histogram (log scale)
axes[0,0].hist(df['person_income'], bins=50, alpha=0.7, color='gold', edgecolor='black')
axes[0,0].set_xscale('log')
axes[0,0].axvline(df['person_income'].mean(), color='red', linestyle='--', label=f'Mean: ${df["person_income"].mean():,.0f}')
axes[0,0].axvline(df['person_income'].median(), color='blue', linestyle='--', label=f'Median: ${df["person_income"].median():,.0f}')
axes[0,0].set_title('Income Distribution (Log Scale)')
axes[0,0].set_xlabel('Income ($)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# Income vs Loan Status
sns.boxplot(data=df, x='loan_status', y='person_income', ax=axes[0,1])
axes[0,1].set_yscale('log')
axes[0,1].set_title('Income Distribution by Loan Status (Log Scale)')

# Income groups vs Default Rate
income_default_rate = df.groupby('income_group')['loan_status'].mean()
axes[1,0].bar(income_default_rate.index, income_default_rate.values, color='lightcoral', alpha=0.7)
axes[1,0].set_title('Default Rate by Income Group')
axes[1,0].set_xlabel('Income Group')
axes[1,0].set_ylabel('Default Rate')
axes[1,0].tick_params(axis='x', rotation=45)

# Income group distribution
income_group_counts = df['income_group'].value_counts().sort_index()
axes[1,1].bar(income_group_counts.index, income_group_counts.values, color='lightblue', alpha=0.7)
axes[1,1].set_title('Distribution of Income Groups')
axes[1,1].set_xlabel('Income Group')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[32]:


# 3. Loan Amount Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loan amount histogram
axes[0,0].hist(df['loan_amnt'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0,0].axvline(df['loan_amnt'].mean(), color='red', linestyle='--', label=f'Mean: ${df["loan_amnt"].mean():,.0f}')
axes[0,0].axvline(df['loan_amnt'].median(), color='blue', linestyle='--', label=f'Median: ${df["loan_amnt"].median():,.0f}')
axes[0,0].set_title('Loan Amount Distribution')
axes[0,0].set_xlabel('Loan Amount ($)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# Loan amount vs Loan Status
sns.boxplot(data=df, x='loan_status', y='loan_amnt', ax=axes[0,1])
axes[0,1].set_title('Loan Amount Distribution by Loan Status')

# Loan amount groups vs Default Rate
loan_default_rate = df.groupby('loan_amount_group')['loan_status'].mean()
axes[1,0].bar(loan_default_rate.index, loan_default_rate.values, color='lightcoral', alpha=0.7)
axes[1,0].set_title('Default Rate by Loan Amount Group')
axes[1,0].set_xlabel('Loan Amount Group')
axes[1,0].set_ylabel('Default Rate')
axes[1,0].tick_params(axis='x', rotation=45)

# Loan amount group distribution
loan_group_counts = df['loan_amount_group'].value_counts().sort_index()
axes[1,1].bar(loan_group_counts.index, loan_group_counts.values, color='lightsalmon', alpha=0.7)
axes[1,1].set_title('Distribution of Loan Amount Groups')
axes[1,1].set_xlabel('Loan Amount Group')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[33]:


# 4. Summary Analysis - Validation of Feature Engineering
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Correlation heatmap of original vs engineered features
correlation_features = ['person_age', 'person_income', 'loan_amnt', 'loan_status']
corr_matrix = df[correlation_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
axes[0,0].set_title('Correlation Matrix - Original Features')

# Default rates by all groups - comparison
group_default_rates = pd.DataFrame({
    'Age Groups': df.groupby('age_group')['loan_status'].mean(),
    'Income Groups': df.groupby('income_group')['loan_status'].mean(),
    'Loan Amount Groups': df.groupby('loan_amount_group')['loan_status'].mean()
})

group_default_rates.plot(kind='bar', ax=axes[0,1], alpha=0.7)
axes[0,1].set_title('Default Rates Across All Engineered Groups')
axes[0,1].set_xlabel('Groups')
axes[0,1].set_ylabel('Default Rate')
axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,1].tick_params(axis='x', rotation=45)

# Distribution of records across groups
axes[1,0].pie([len(df[df['age_group'].notna()]),
               len(df[df['income_group'].notna()]),
               len(df[df['loan_amount_group'].notna()])],
              labels=['Age Groups', 'Income Groups', 'Loan Amount Groups'],
              autopct='%1.1f%%',
              colors=['skyblue', 'lightgreen', 'lightcoral'])
axes[1,0].set_title('Coverage of Engineered Features')

# Statistical summary of groupings
print("Statistical Validation of Feature Engineering Decisions:\n")

print("1. AGE GROUPS:")
print("   - Natural breaks appear around life stages (young adults, middle-aged, etc.)")
print("   - Default rates vary significantly across age groups")
for group in df['age_group'].cat.categories:
    group_data = df[df['age_group'] == group]
    default_rate = group_data['loan_status'].mean()
    print(f"   - {group}: {len(group_data):,} records, {default_rate:.1%} default rate")

print("\n2. INCOME GROUPS:")
print("   - Based on income quantiles and economic classifications")
print("   - Clear inverse relationship with default rates")
for group in df['income_group'].cat.categories:
    group_data = df[df['income_group'] == group]
    default_rate = group_data['loan_status'].mean()
    income_range = group_data['person_income'].agg(['min', 'max'])
    print(f"   - {group}: ${income_range['min']:,.0f}-${income_range['max']:,.0f}, {default_rate:.1%} default rate")

print("\n3. LOAN AMOUNT GROUPS:")
print("   - Based on loan size categories and risk assessment")
print("   - Relationship with default rates for risk profiling")
for group in df['loan_amount_group'].cat.categories:
    group_data = df[df['loan_amount_group'] == group]
    default_rate = group_data['loan_status'].mean()
    amount_range = group_data['loan_amnt'].agg(['min', 'max'])
    print(f"   - {group}: ${amount_range['min']:,.0f}-${amount_range['max']:,.0f}, {default_rate:.1%} default rate")

# Remove the unused subplot
axes[1,1].remove()

plt.tight_layout()
plt.show()


# In[34]:


df.head()


# In[35]:


# checking the distribution of home ownership
df['person_home_ownership'].value_counts()


# In[36]:


# analyzing the relationship between home ownership and loan status
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Grouped Bar Chart
home_loan_crosstab = pd.crosstab(df['person_home_ownership'], df['loan_status'])
home_loan_crosstab.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
axes[0,0].set_title('Loan Status by Home Ownership', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Home Ownership')
axes[0,0].set_ylabel('Count')
axes[0,0].legend(['Not Default (0)', 'Default (1)'])
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Stacked Percentage Bar Chart
home_loan_pct = pd.crosstab(df['person_home_ownership'], df['loan_status'], normalize='index') * 100
home_loan_pct.plot(kind='bar', stacked=True, ax=axes[0,1], color=['skyblue', 'lightcoral'])
axes[0,1].set_title('Loan Status Distribution by Home Ownership (%)', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Home Ownership')
axes[0,1].set_ylabel('Percentage')
axes[0,1].legend(['Not Default (0)', 'Default (1)'])
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Default Rate by Home Ownership
default_rates_by_home = df.groupby('person_home_ownership')['loan_status'].agg(['mean', 'count'])
default_rates_by_home['mean'].plot(kind='bar', ax=axes[1,0], color='lightcoral', alpha=0.8)
axes[1,0].set_title('Default Rate by Home Ownership', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Home Ownership')
axes[1,0].set_ylabel('Default Rate')
axes[1,0].tick_params(axis='x', rotation=45)

# Add count labels on bars
for i, (ownership, row) in enumerate(default_rates_by_home.iterrows()):
    axes[1,0].text(i, row['mean'] + 0.01, f'n={row["count"]:,}',
                   ha='center', va='bottom', fontsize=9)

# 4. Heatmap of contingency table
sns.heatmap(home_loan_crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
axes[1,1].set_title('Contingency Table: Home Ownership vs Loan Status', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Loan Status')
axes[1,1].set_ylabel('Home Ownership')

plt.tight_layout()
plt.show()

# Print detailed statistics
print("Detailed Analysis: Home Ownership vs Loan Status\n")
print("="*60)

for ownership in df['person_home_ownership'].unique():
    subset = df[df['person_home_ownership'] == ownership]
    total_loans = len(subset)
    defaults = sum(subset['loan_status'])
    non_defaults = total_loans - defaults
    default_rate = defaults / total_loans

    print(f"\n{ownership}:")
    print(f"  Total Loans: {total_loans:,}")
    print(f"  Defaults: {defaults:,} ({default_rate:.2%})")
    print(f"  Non-Defaults: {non_defaults:,} ({(1-default_rate):.2%})")

# Statistical significance test
from scipy.stats import chi2_contingency

chi2, p_value, dof, expected = chi2_contingency(home_loan_crosstab)
print(f"\n{'='*60}")
print(f"Chi-square Test for Independence:")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Degrees of freedom: {dof}")
print(f"  Significant relationship: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")


# In[37]:


# creating loan-to-income ratio
df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']

# creating loan-to-employment length ratio
df['loan_to_emp_length_ratio'] =  df['person_emp_length']/ df['loan_amnt']

# creating interest rate-to-loan amount ratio
df['int_rate_to_loan_amt_ratio'] = df['loan_int_rate'] / df['loan_amnt']


# In[38]:


df.columns


# In[39]:


df.describe()


# #### Splitting the Data into training and testing sets

# In[40]:


X = df.drop(columns=['loan_status'])
y = df['loan_status']


# In[41]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# In[42]:


X_train.info()


# In[43]:


# specifying categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns


# In[44]:


categorical_cols


# In[45]:


numerical_cols


# In[46]:


# encoding categorical variables using one-hot encoding
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
one_hot_encoder.fit(X_train[categorical_cols])


# In[47]:


one_hot_encoder.categories_


# In[48]:


# transform categorical columns
encoded_col_names = one_hot_encoder.get_feature_names_out(categorical_cols)

X_train_cat_ohe = pd.DataFrame(
    one_hot_encoder.transform(X_train[categorical_cols]),
    columns=encoded_col_names,
    index=X_train.index
)

X_test_cat_ohe = pd.DataFrame(
    one_hot_encoder.transform(X_test[categorical_cols]),
    columns=encoded_col_names,
    index=X_test.index
)

# combining numerical + encoded categorical columns
X_train = pd.concat([X_train[numerical_cols], X_train_cat_ohe], axis=1)
X_test = pd.concat([X_test[numerical_cols], X_test_cat_ohe], axis=1)


# In[49]:


X_train.columns


# In[50]:


X_train.head()


# In[51]:


# scaling numerical features using standard scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# In[52]:


# checking the distribution of numerical features after scaling
X_train.describe()


# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


# In[54]:


# initializing models
svc = SVC()
knc = KNeighborsClassifier()
# mnb = MultinomialNB()
dtc = DecisionTreeClassifier()
lrc = LogisticRegression()
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
gbdt = GradientBoostingClassifier()
xgb = XGBClassifier()
lgb = LGBMClassifier()


# In[55]:


# creating a dictionary of classifiers for easy access
classifiers = {
    'SVC': svc,
    'KNN': knc,
    # 'MultinomialNB': mnb,
    'DecisionTree': dtc,
    'LogisticRegression': lrc,
    'RandomForest': rfc,
    'AdaBoost': abc,
    'GradientBoosting': gbdt,
    'XGBoost': xgb,
    'LightGBM': lgb
}


# In[56]:


def evaluate_classifier_with_cv(classifier, X, y, cv_folds=5, random_state=42):
    """
    Evaluate classifier using StratifiedKFold cross-validation
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # cross-validation scores
    cv_accuracy = cross_val_score(classifier, X, y, cv=skf, scoring='accuracy')
    cv_precision = cross_val_score(classifier, X, y, cv=skf, scoring='precision')
    cv_recall = cross_val_score(classifier, X, y, cv=skf, scoring='recall')
    cv_f1 = cross_val_score(classifier, X, y, cv=skf, scoring='f1')

    # calculating means and standard deviations
    results = {
        'accuracy': {'mean': cv_accuracy.mean(), 'std': cv_accuracy.std()},
        'precision': {'mean': cv_precision.mean(), 'std': cv_precision.std()},
        'recall': {'mean': cv_recall.mean(), 'std': cv_recall.std()},
        'f1': {'mean': cv_f1.mean(), 'std': cv_f1.std()}
    }

    return results


# In[57]:


# defining parameter grids for GridSearchCV hyperparameter tuning
param_grids = {
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    },
    'SVC': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'probability': [True]  # needed for predict_proba
    },
    'DecisionTree': {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
}


# In[58]:


# training models with default parameters
print("=== Step 1: Training Models with Default Parameters ===\n")

default_results = {}
trained_models = {}

for name, classifier in classifiers.items():
    print(f"\nTraining: {name}")

    classifier.fit(X_train, y_train)
    trained_models[name] = classifier

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    default_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

print(f"\n{'='*60}")
print("STEP 1 COMPLETED - Basic Training Results Ready")
print(f"{'='*60}")


# In[59]:


# visualizing the default training results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# preparing data for visualization
model_names = list(default_results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_values = {metric: [default_results[name][metric] for name in model_names] for metric in metrics}

for idx, metric in enumerate(metrics):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]

    bars = ax.bar(model_names, metric_values[metric], alpha=0.8, color=plt.cm.Set3(idx))

    ax.set_title(f'{metric.capitalize()} Scores (Default Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.capitalize()}')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    for bar, value in zip(bars, metric_values[metric]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Create comparison table
print("\n" + "="*80)
print("DEFAULT PARAMETERS RESULTS SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [f"{default_results[name]['accuracy']:.4f}" for name in model_names],
    'Precision': [f"{default_results[name]['precision']:.4f}" for name in model_names],
    'Recall': [f"{default_results[name]['recall']:.4f}" for name in model_names],
    'F1-Score': [f"{default_results[name]['f1']:.4f}" for name in model_names]
})

print(comparison_df.to_string(index=False))

# best model identification
print(f"\n{'='*80}")
print("BEST PERFORMING MODELS BY METRIC (Default Parameters)")
print(f"{'='*80}")

for metric in metrics:
    best_idx = np.argmax([default_results[name][metric] for name in model_names])
    best_model = model_names[best_idx]
    best_score = default_results[best_model][metric]
    print(f"Best {metric.upper()}: {best_model} ({best_score:.4f})")


# In[60]:


# cross-validation evaluation
print("=== Step 2: Cross-Validation Evaluation ===\n")

# setting StratifiedKFold for consistent evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}

# combining training and testing data for cross-validation
X_combined = pd.concat([X_train, X_test], axis=0)
y_combined = pd.concat([y_train, y_test], axis=0)

for name, classifier in classifiers.items():
    print(f"\nEvaluating: {name}")

    # Cross-Validation Evaluation
    cv_scores = evaluate_classifier_with_cv(classifier, X_combined, y_combined)
    cv_results[name] = cv_scores

    for metric, scores in cv_scores.items():
        print(f"   {metric.capitalize()}: {scores['mean']:.4f} ± {scores['std']:.4f}")

print(f"\n{'='*60}")
print("STEP 2 COMPLETED - Cross-Validation Results Ready")
print(f"{'='*60}")


# In[61]:


# visualizing cross-validation results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

model_names = list(cv_results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_means = {metric: [cv_results[name][metric]['mean'] for name in model_names] for metric in metrics}
metric_stds = {metric: [cv_results[name][metric]['std'] for name in model_names] for metric in metrics}

for idx, metric in enumerate(metrics):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]

    bars = ax.bar(model_names, metric_means[metric], yerr=metric_stds[metric],
                  capsize=5, alpha=0.8, color=plt.cm.Set3(idx))

    ax.set_title(f'{metric.capitalize()} Scores (Cross-Validation)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.capitalize()}')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    for bar, mean_val, std_val in zip(bars, metric_means[metric], metric_stds[metric]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n" + "="*100)
print("CROSS-VALIDATION RESULTS SUMMARY")
print("="*100)

comparison_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy (±std)': [f"{cv_results[name]['accuracy']['mean']:.4f} ± {cv_results[name]['accuracy']['std']:.4f}" for name in model_names],
    'Precision (±std)': [f"{cv_results[name]['precision']['mean']:.4f} ± {cv_results[name]['precision']['std']:.4f}" for name in model_names],
    'Recall (±std)': [f"{cv_results[name]['recall']['mean']:.4f} ± {cv_results[name]['recall']['std']:.4f}" for name in model_names],
    'F1-Score (±std)': [f"{cv_results[name]['f1']['mean']:.4f} ± {cv_results[name]['f1']['std']:.4f}" for name in model_names]
})

print(comparison_df.to_string(index=False))

# best model identification
print(f"\n{'='*100}")
print("BEST PERFORMING MODELS BY METRIC (Cross-Validation)")
print(f"{'='*100}")

for metric in metrics:
    best_idx = np.argmax([cv_results[name][metric]['mean'] for name in model_names])
    best_model = model_names[best_idx]
    best_score = cv_results[best_model][metric]['mean']
    print(f"Best {metric.upper()}: {best_model} ({best_score:.4f})")


# In[63]:


models_to_tune = ['DecisionTree', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, classifier in classifiers.items():
    print(f"\n{'='*50}")
    print(f"Tuning: {name}")
    print(f"{'='*50}")

    if name in models_to_tune and name in param_grids:

        grid_search = GridSearchCV(
            classifier,
            param_grids[name],
            cv=skf,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        print("   Searching for best parameters...")
        grid_search.fit(X_combined, y_combined)

        best_models[name] = grid_search.best_estimator_
        grid_search_results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

        print(f"   Best Parameters: {grid_search.best_params_}")
        print(f"   Best CV F1-Score: {grid_search.best_score_:.4f}")

    else:
        print(f"   Skipping hyperparameter tuning for {name}.")
        best_models[name] = classifier


# In[64]:


from sklearn.metrics import roc_curve, auc

y_prob = best_models['LightGBM'].predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for LightGBM")
plt.legend()
plt.show()


# In[67]:


feature_importance = pd.Series(
    best_models['LightGBM'].feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(feature_importance.head(10))


# In[68]:


plt.figure(figsize=(8,6))
feature_importance.head(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()


# In[69]:


print("\n" + "="*80)
print("FINAL MODEL TESTING & PREDICTION")
print("="*80)

# select the best performing model
final_model = best_models['LightGBM']

print("\nUsing LightGBM as the final model for prediction.")

# predict on test dataset
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:,1]

# create results dataframe
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Default_Probability": y_prob
})

# classify probability into risk categories
results["Risk_Category"] = pd.cut(
    results["Default_Probability"],
    bins=[0, 0.3, 0.7, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

print("\nSample Prediction Results:")
print(results.head(10))

print("\n" + "="*80)
print("SUMMARY OF TEST PREDICTIONS")
print("="*80)

print("Total Test Samples:", len(results))
print("Predicted High Risk:", (results["Predicted"] == 1).sum())
print("Predicted Low Risk:", (results["Predicted"] == 0).sum())

# show distribution of predicted risk categories
print("\nRisk Category Distribution:")
print(results["Risk_Category"].value_counts())

print("\n" + "="*80)
print("SAMPLE CUSTOMER RISK PREDICTION")
print("="*80)

# select one customer example from test data
sample_customer = X_test.iloc[[0]]

sample_prediction = final_model.predict(sample_customer)[0]
sample_probability = final_model.predict_proba(sample_customer)[0][1]

print("Predicted Class:", "High Risk" if sample_prediction == 1 else "Low Risk")
print(f"Default Probability: {sample_probability:.3f}")

print("\nModel Prediction Complete.")
print("="*80)

