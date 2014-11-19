import os

from .data_model.study import Study
import compute
from compute.predict import PredictorConfigManager, PredictorDataSetManager
import data_model
from flotilla.datapackage import make_study_datapackage
import visualize

__version__ = '0.2.4dev'

_neural_diff_chr22 = 'http://sauron.ucsd.edu/flotilla_projects/' \
                     'neural_diff_chr22/datapackage.json'
_shalek2013 = 'https://raw.githubusercontent.com/YeoLab/shalek2013/master/' \
              'datapackage.json'

def embark(study_name, load_species_data=True):
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
            return Study.from_datapackage_file(study_name,
                                               load_species_data=load_species_data)
        except IOError:
            pass
        filename = os.path.abspath(os.path.expanduser(
            '~/flotilla_projects/{}/datapackage.json'.format(study_name)))
        import pdb; pdb.set_trace()
        return Study.from_datapackage_file(filename,
                                           load_species_data=load_species_data)
    except IOError:
        return Study.from_datapackage_url(study_name,
                                          load_species_data=load_species_data)
