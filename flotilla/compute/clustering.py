import fastcluster
from scipy.spatial import distance


class Cluster(object):
    def __call__(self, data, metric, linkage_method):
        row_linkage = fastcluster.linkage(data.values, method=linkage_method,
                                          metric=metric)
        col_linkage = fastcluster.linkage(data.values.T, method=linkage_method,
                                          metric=metric)
        # row_linkage = self._linkage(self._pairwise_dists(data.values,
        #                                                  axis=0,
        #                                                  metric=metric),
        #                             linkage_method)
        #
        # col_linkage = self._linkage(self._pairwise_dists(data.values,
        #                                                  axis=1,
        #                                                  metric=metric),
        #                             linkage_method)
        return row_linkage, col_linkage

    @staticmethod
    def _pairwise_dists(values, axis=0, metric='euclidean'):
        """
        Parameters
        ----------
        values : numpy.array
            The array you want to calculate pairwise distances for
        axis : int
            Which axis to calculate the pairwise distances for.
            0: pairwise distances between all rows
            1: pairwise distances between all columns
            Default 0
        metric : str
            Which distance metric to use. Default 'euclidean'

        Returns
        -------


        Raises
        ------
        """
        if axis == 1:
            values = values.T
        return distance.squareform(distance.pdist(values,
                                                  metric=metric))

    @staticmethod
    def _linkage(pairwise_dists, linkage_method):
        return fastcluster.linkage_vector(pairwise_dists,
                                          method=linkage_method)
