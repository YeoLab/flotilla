import os

from .data_model import ExpressionData, SplicingData, MetaData, \
    MappingStatsData, GeneOntologyData
from .compute.predict import PredictorConfigManager, PredictorDataSetManager
from .datapackage import make_study_datapackage, FLOTILLA_DOWNLOAD_DIR
from .study import Study


__all__ = ['Study', 'PredictorConfigManager', 'PredictorDataSetManager',
           'make_study_datapackage', 'FLOTILLA_DOWNLOAD_DIR',
           'compute', 'data_model', 'visualize', 'Study', 'ExpressionData',
           'SplicingData', 'MetaData', 'MappingStatsData',
           'datapackage', 'GeneOntologyData', 'go', 'util']

__version__ = 'v0.3.0'

# 18 cells, multiindex on the splicing data features, features already renamed
# in the matrices
_shalek2013 = 'https://raw.githubusercontent.com/YeoLab/shalek2013/master/' \
              'datapackage.json'

# 250 cells, ensembl and miso ids on index, need renaming, lots of celltypes
_test_data = 'https://raw.githubusercontent.com/YeoLab/flotilla_test_data/' \
             'master/datapackage.json'
_brainspan = 'https://s3-us-west-2.amazonaws.com/flotilla/' \
             'brainspan_batch_corrected_for_amazon_s3/datapackage.json'


def embark(study_name, load_species_data=True,
           flotilla_dir=FLOTILLA_DOWNLOAD_DIR):
    """
    Begin your journey of data exploration.

    Parameters
    ----------
    data_package_url : str
        A URL to a datapackage.json file

    Returns
    -------
    study : flotilla.Study
        A biological study created from the data package specified
    """
    try:
        try:
            return Study.from_datapackage_file(
                study_name, load_species_data=load_species_data)
        except IOError:
            pass
        filename = os.path.abspath(os.path.expanduser(
            '{}/{}/datapackage.json'.format(flotilla_dir,
                                            study_name)))
        return Study.from_datapackage_file(filename,
                                           load_species_data=load_species_data)
    except IOError:
        return Study.from_datapackage_url(study_name,
                                          load_species_data=load_species_data)
