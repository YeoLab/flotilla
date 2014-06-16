__author__ = 'olga'

from _ExpressionData import ExpressionData

class SpikeInData(ExpressionData):
    """Class for Spikein data and associated functions
    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, df, sample_metadata):
        """Constructor for

        Parameters
        ----------
        df, sample_metadata

        Returns
        -------


        Raises
        ------

        """
        pass

    def spikeins_violinplot(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        fig, axes = plt.subplots(nrows=5, figsize=(16, 20), sharex=True,
                                 sharey=True)
        ercc_concentrations = ercc_controls_analysis.mix1_molecules_per_ul.copy()
        ercc_concentrations.sort()

        for ax, (celltype, celltype_df) in zip(axes.flat,
                                               tpm.ix[spikeins].groupby(
                                                       sample_id_to_celltype_,
                                                       axis=1)):
            print celltype
            #     fig, ax = plt.subplots(figsize=(16, 4))
            x_so_far = 0
            #     ax.set_yscale('log')
            xticklabels = []
            for spikein_type, spikein_df in celltype_df.groupby(
                    spikein_to_type):
                #         print spikein_df.shape
                df = spikein_df.T + np.random.uniform(0, 0.01,
                                                      size=spikein_df.T.shape)
                df = np.log2(df)
                if spikein_type == 'ERCC':
                    df = df[ercc_concentrations.index]
                xticklabels.extend(df.columns.tolist())
                color = 'husl' if spikein_type == 'ERCC' else 'Greys_d'
                sns.violinplot(df, ax=ax,
                               positions=np.arange(df.shape[1]) + x_so_far,
                               linewidth=0, inner='none', color=color)

                x_so_far += df.shape[1]

            ax.set_title(celltype)
            ax.set_xticks(np.arange(x_so_far))
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
            ax.set_ylabel('$\\log_2$ TPM')

            xmin, xmax = -0.5, x_so_far - 0.5

            ax.hlines(0, xmin, xmax)
            ax.set_xlim(xmin, xmax)
            sns.despine()
            # fig.savefig('/projects/ps-yeolab/obotvinnik/mn_diff_singlecell/figures/spikeins.pdf')
            # ! cp /projects/ps-yeolab/obotvinnik/mn_diff_singlecell/figures/spikeins.pdf ~/Dropbox/figures2/singlecell/spikeins.pdf

        def samples_violinplot():
            fig, axes = plt.subplots(nrows=3, figsize=(16, 6))

            for ax, (spikein_type, df) in zip(axes, tpm.groupby(spikein_to_type,
                                                                axis=0)):
                print spikein_type, df.shape

                if df.shape[0] > 1:
                    sns.violinplot(np.log2(df + 1), ax=ax, linewidth=0.1)
                    ax.set_xticks([])
                    ax.set_xlabel('')

                else:
                    x = np.arange(df.shape[1])
                    ax.bar(np.arange(df.shape[1]), np.log2(df.ix[spikein_type]),
                           color=green)
                    ax.set_xticks(x + 0.4)
                    ax.set_xticklabels(df.columns, rotation=60)
                    sns.despine()

                ax.set_title(spikein_type)
                ax.set_xlim(0, tpm.shape[1])
                ax.set_ylabel('$\\log_2$ TPM')
            sns.despine()