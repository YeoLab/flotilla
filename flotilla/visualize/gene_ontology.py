import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_go_enrichment(go, ax=None):
    go.bonferonni_corrected_p_value = go.bonferonni_corrected_p_value.replace(0, np.nan)
    vmin = max(go.bonferonni_corrected_p_value.dropna().min(), 1e-25)
    if np.isnan(vmin):
        vmin = 1e-25
    go.loc[:, 'bonferonni_corrected_p_value'] = go.bonferonni_corrected_p_value.fillna(vmin*.9)
    if go.shape[0] > 20:
        go = go.iloc[-20: , :]
    go_subset = go
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(go_subset.shape[0]*.25, 4)))
    else:
        ax = plt.gca()
        fig = plt.gcf()

    bottom = np.arange(go_subset.shape[0])
    width = -np.log10(go_subset.bonferonni_corrected_p_value)
    ax.barh(bottom, width)
    xticks = list(sorted(list(set(int(x) for x in ax.get_xticks()))))
    ax.set(yticks=bottom+0.4, yticklabels=go_subset.go_name, xlabel='$-\log_{10} p$-value',
           ylim=(0, bottom.max()+1))
    sns.despine()
    fig.tight_layout()
    return ax