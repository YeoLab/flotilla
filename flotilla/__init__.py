__author__ = 'lovci'

from .data_model.study import StudyFactory, Study


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
    """
    # TODO: This
    return Study.from_data_package_url(data_package_url)
