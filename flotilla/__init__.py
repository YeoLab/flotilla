import os

from .data_model.study import Study
import compute
import data_model
from .external import make_study_datapackage
import visualize

try:
    get_ipython().magic(u'matplotlib inline')
except:
    pass


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
        return Study.from_datapackage_file(filename,
                                           load_species_data=load_species_data)
    except IOError:
        return Study.from_datapackage_url(study_name,
                                          load_species_data=load_species_data)
