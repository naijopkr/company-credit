import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/loan_data.csv')
df.head()
df.info()
df.describe()

sns.set_style('whitegrid')

def plot_hist():
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


def countplot():
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

def lmplot():
    sns.lmplot(
        x='fico',
        y='int.rate',
        col='not.fully.paid',
        data=df,
        hue='credit.policy',
        palette='coolwarm'
    )


plot_hist()

countplot()

sns.jointplot(x='fico', y='int.rate', data=df, color='purple')

lmplot()
