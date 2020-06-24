# -*- coding: utf-8 -*-


import h5py
import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing

def creazione_df(nobs, nvar, dim_segnale, sigmas, flat_frac):
    sigma = np.random.uniform(sigmas[0], sigmas[1], dim_segnale)
    mean = np.random.uniform(3*sigma, 1-3*sigma)
    gauss_indices = random.sample(range(nvar), dim_segnale)

    columns = list(np.arange(nvar))
    index = list(np.arange(nobs))

    df = pd.DataFrame(np.random.uniform(0, 1, size=(nobs, nvar)), columns=columns)
    df.insert(0, "Signal", np.zeros(nobs), True)
    for i in np.arange(1,nobs):
        tmp = 0
        if i < flat_frac*nobs:
            df.iloc[i, 0] = 0
        else:
            for p in range(nvar):
                if p in gauss_indices:
                    df.iloc[i,p] = np.random.normal(mean[tmp], sigma[tmp])
                    df.iloc[i, 0] = 1
                    tmp += 1
    df = df.sample(frac=1).reset_index(drop=True)

    return df


if __name__ == "__main__":

    df0 = creazione_df(10000, 20, 15, np.array([0.01, 0.1]), 0.95)
    df0.to_csv('dati_generati_originali_5perc.csv', index=False)

    df1 = creazione_df(10000, 20, 15, np.array([0.01, 0.1]), 0.98)
    df1.to_csv('dati_generati_originali_2perc.csv', index=False)

    df2 = creazione_df(10000, 20, 15, np.array([0.08, 0.1]), 0.95)
    df2.to_csv('dati_generati_var_maggiori_5perc.csv', index=False)

    df3 = creazione_df(10000, 20, 15, np.array([0.08, 0.1]), 0.98)
    df3.to_csv('dati_generati_var_maggiori_2perc.csv', index=False)

    df4 = creazione_df(10000, 20, 17, np.array([0.01, 0.1]), 0.95)
    df4.to_csv('dati_generati_17_dim_5perc.csv', index=False)

    df5 = creazione_df(10000, 20, 13, np.array([0.01, 0.1]), 0.95)
    df5.to_csv('dati_generati_13_dim_5perc.csv', index=False)

    df6 = creazione_df(10000, 20, 17, np.array([0.01, 0.1]), 0.98)
    df6.to_csv('dati_generati_17_dim_2perc.csv', index=False)

    df7 = creazione_df(10000, 20, 13, np.array([0.01, 0.1]), 0.98)
    df7.to_csv('dati_generati_13_dim_2perc.csv', index=False)

    df8 = creazione_df(10000, 20, 15, np.array([0.01, 0.05]), 0.95)
    df8.to_csv('dati_generati_var_minori_5perc.csv', index=False)

    df9 = creazione_df(10000, 20, 15, np.array([0.01, 0.05]), 0.98)
    df9.to_csv('dati_generati_var_minori_2perc.csv', index=False)

    df10 = creazione_df(10000, 20, 15, np.array([0.01, 0.1]), 0.92)
    df10.to_csv('dati_generati_originali_8perc.csv', index=False)