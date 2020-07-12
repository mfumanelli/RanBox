import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import random


def creazione_df(nobs, nvar, dim_segnale, sigmas, flat_frac):
    sigma = np.random.uniform(sigmas[0], sigmas[1], dim_segnale)
    mean = np.random.uniform(3 * sigma, 1 - (3 * sigma))
    gauss_indices = random.sample(range(1, nvar), dim_segnale)

    columns = list(np.arange(nvar))

    df = pd.DataFrame(np.random.uniform(0, 1, size=(nobs, nvar)), columns=columns)
    df.insert(0, "Signal", np.zeros(nobs), True)
    for i in np.arange(0, nobs):
        tmp = 0
        if i < flat_frac * nobs:
            df.iloc[i, 0] = 0
        else:
            for p in gauss_indices:
                df.iloc[i, p] = np.random.normal(mean[tmp], sigma[tmp])
                df.iloc[i, 0] = 1
                tmp += 1
    df = df.sample(frac=1).reset_index(drop=True)

    return df


if __name__ == "__main__":

    # df = creazione_df(10000, 20, 15, np.array([0.01, 0.1]), 0.99)
    df = pd.read_csv(r"dati_generati/dati_frodi_def_1perc.csv")
    columns = df.drop("Signal", axis=1).columns
    grid = gridspec.GridSpec(6, 5)

    # GRAFICO VARIABILE RISPOSTA
    count_classes = pd.value_counts(df["Signal"], sort=True)
    count_classes.plot(kind="bar", rot=0)
    plt.title("Signal Distribution")
    # plt.xticks(range(2), LABELS)
    plt.xlabel("Signal")
    plt.ylabel("Frequency")

    plt.figure(figsize=(20, 10 * 2))

    # GRAFICO DISTRIBUZIONE FEATURES SEGNALE (rosso) VS NON SEGNALE (verde)
    for n, col in enumerate(df[columns]):
        ax = plt.subplot(grid[n])
        sns.distplot(df[df.Signal == 1][col], bins=50, color="r")
        sns.distplot(df[df.Signal == 0][col], bins=50, color="g")
        ax.set_ylabel("Density")
        ax.set_title(str(col))
        ax.set_xlabel("")

    plt.show()
