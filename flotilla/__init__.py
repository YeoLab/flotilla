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

# def call_main():
#     # where is this "agrparser" initialized?
#     args = argparser.parse_args()
#     params = vars(args)
#     main(params)


def main():
    pass


def embark(study_name):
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
        filename = os.path.abspath(os.path.expanduser(
            '~/flotilla_projects/{}/datapackage.json'.format(study_name)))
        return Study.from_data_package_file(filename)
    except:
        return Study.from_data_package_url(study_name)
