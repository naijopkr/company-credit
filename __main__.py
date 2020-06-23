import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/loan_data.csv')
df.head()
df.info()
df.describe()

sns.set_style('whitegrid')

def hist_credit_policy():
    plt.figure(figsize=(10, 6))
    df[df['credit.policy'] == 1]['fico'].hist(
        bins=30,
        color='blue',
        alpha=0.5,
        label='Credit Policy = 1'
    )
    df[df['credit.policy'] == 0]['fico'].hist(
        bins=30,
        color='red',
        alpha=0.5,
        label='Credit Policy = 0'
    )
    plt.legend()
    plt.xlabel('FICO')


def count_purpose():
    plt.figure(figsize=(10, 10))
    ax = sns.countplot(
        x=df['purpose'],
        hue=df['not.fully.paid']
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=40,
        ha='right'
    )
    plt.tight_layout()

def joint_fico_intrate():
    sns.jointplot(x='fico', y='int.rate', data=df, color='purple')


def lm_fico_intrate():
    sns.lmplot(
        x='fico',
        y='int.rate',
        col='not.fully.paid',
        data=df,
        hue='credit.policy',
        palette='coolwarm'
    )

hist_credit_policy()

count_purpose()

joint_fico_intrate()

lm_fico_intrate()

# Setting up the data

# Categorigal features
cat_feats = ['purpose']
final_data = pd.get_dummies(df, columns=cat_feats, drop_first=True)
final_data.head()

# Train test split
X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# Training Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_dtree_pred = dtree.predict(X_test)

# Evaluate decision tree
from sklearn.metrics import classification_report, confusion_matrix
from utils import print_cm

print_cm(confusion_matrix(y_test, y_dtree_pred))
print(classification_report(y_test, y_dtree_pred))


# Training Random Forest model
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

y_rfc_pred = rfc.predict(X_test)

# Evaluate RFC
print_cm(confusion_matrix(y_test, y_rfc_pred))
print(classification_report(y_test, y_rfc_pred))

# Print decision tree image
from IPython.display import Image
from io import StringIO
from sklearn.tree import export_graphviz
import pydot

features = list(final_data.columns[1:])
features

dot_data = StringIO()
export_graphviz(
    dtree,
    out_file=dot_data,
    feature_names=features,
    filled=True,
    rounded=True
)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())
