# import fastcluster
# from scipy.spatial import distance
#
#
# class Cluster(object):
#     def __call__(self, data, metric, linkage_method):
#         row_linkage = fastcluster.linkage(data.values, method=linkage_method,
#                                           metric=metric)
#         col_linkage = fastcluster.linkage(data.values.T, method=linkage_method,
#                                           metric=metric)
#         # row_linkage = self._linkage(self._pairwise_dists(data.values,
#         #                                                  axis=0,
#         #                                                  metric=metric),
#         #                             linkage_method)
#         #
#         # col_linkage = self._linkage(self._pairwise_dists(data.values,
#         #                                                  axis=1,
#         #                                                  metric=metric),
#         #                             linkage_method)
#         return row_linkage, col_linkage
#
#     @staticmethod
#     def _pairwise_dists(values, axis=0, metric='euclidean'):
#         """
#         Parameters
#         ----------
#         values : numpy.array
#             The array you want to calculate pairwise distances for
#         axis : int
#             Which axis to calculate the pairwise distances for.
#             0: pairwise distances between all rows
#             1: pairwise distances between all columns
#             Default 0
#         metric : str
#             Which distance metric to use. Default 'euclidean'
#
#         Returns
#         -------
#
#
#         Raises
#         ------
#         """
#         if axis == 1:
#             values = values.T
#         return distance.squareform(distance.pdist(values,
#                                                   metric=metric))
#
#     @staticmethod
#     def _linkage(pairwise_dists, linkage_method):
#         return fastcluster.linkage_vector(pairwise_dists,
#                                           method=linkage_method)
#
# import numpy as np
# import tempfile
# import subprocess
# import glob
# class OLO(object):
#     raise NotImplementedError
#     # Z Bar-Joseph Optimal Leaf Ordering (OLO) executable (zip is already in this repo)
#     #
#     # download from here: http://www.cs.cmu.edu/~zivbj/Optimal.zip
#
#     #try:
#     #    subprocess.check_call([oloCMD, olotestFile])
#
#     #except:
#     #    if not os.path.exists(pth + "Optimal.zip"):
#     #        raise Exception
#
#         #unzipme = ZipFile(pth + "Optimal.zip")
#
#         #x = unzipme.extractall(pth)
#
#         #os.chdir(pth + "Optimal/")
#         #subprocess.call(["pwd"])
#         #subprocess.call(["make"])
#         #os.chdir("..")
#
#
#     def parse_CDT(CDTFilename):
#         """Get dendrogram leaf ranks from CDT file."""
#         with open(CDTFilename, 'r') as f:
#             f.readline()
#             f.readline()
#             f.readline()#3 header lines
#             data = f.readlines()
#
#         ordered =np.ndarray(shape=(len(data), 1), dtype='i8')
#         for i, line in enumerate(data):
#             fields = line.strip().split("\t")
#             orig = fields[0]
#             origRank = int(orig.lstrip("GENE").rstrip("X"))-1
#             ordered[origRank] = i
#         return ordered
#
#     def format_for_OLO(df):
#         """
#         Generate a matrix with data row/col labels.
#         """
#         data, rowLabels, colLabels = df.values, df.index, df.columns
#         nRow, nCol = df.shape
#         t = np.vstack([np.insert(colLabels, 0, "gene"), np.column_stack([rowLabels, data])])
#         return t
#
#     def run_OLO(dataFileName, oloCMD = oloCMD):
#         """
#         Call Z Bar-Joseph optimal leaf ordering algoithm on a file with row and col labels
#         http://www.ncbi.nlm.nih.gov/pubmed/11472989
#         """
#         if dataFileName.endswith(".txt"):
#             dataFileName = dataFileName.rstrip(".txt")
#         if not os.path.exists(dataFileName + ".txt"):
#             raise Exception
#         exitStatus = subprocess.check_call([oloCMD, dataFileName])
#         if not exitStatus == 0:
#             raise Exception
#         return
#
#
#     def get_OLO(dataMatrix, rowLabels, colLabels):
#         """
#         given a data matrix with row AND col labels, run optimal leaf ordering
#         return OLO exit status, CDT array, GTR array
#         remove intermediate files
#         """
#         dataTable = format_for_OLO(dataMatrix, rowLabels, colLabels)
#         tmpfileFH, tmpfile = tempfile.mkstemp(dir="./")
#         txtfile = tmpfile + ".txt"
#         save_txt(dataTable, txtfile)
#         run_OLO(tmpfile)
#         ranks = parse_CDT(tmpfile + ".CDT")
#         #GTR = parse_GTR(tmpfile)
#         for fi in glob.glob("%s*"%(tmpfile)):
#             try:
#                 os.remove(fi)
#             except:
#                 raise
#         return ranks