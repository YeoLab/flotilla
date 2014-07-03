from .data_model.study import Study
import compute
import data_model
from .external import make_study_datapackage
import visualize


def call_main():
    # where is this "agrparser" initialized?
    args = argparser.parse_args()
    params = vars(args)
    main(params)


def main():
    pass


def embark(data_package_url):
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
    return Study.from_data_package_url(data_package_url)
