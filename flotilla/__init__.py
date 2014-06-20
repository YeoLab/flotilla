__author__ = 'lovci'

import urllib2
import json

from .data_model.study import StudyFactory, Study


def call_main():

    # where is this "agrparser" initialized?
    args = argparser.parse_args()
    params = vars(args)
    main(params)

def main():
    pass


class Embark(object):
    """
    Begin your journey of data exploration
    """
    def __init__(self, data_package_url):
        self.data_package_url = data_package_url
        self.data_package = self.fetch_data_package()

    def fetch_data_package(self):
        req = urllib2.Request(self.data_package_url)
        opener = urllib2.build_opener()
        f = opener.open(req)
        return json.load(f)

    def create_package(self):
        """

        """
        # Seems contrived that we're instantiating StudyFactory and just
        # using one method from it..
        study = StudyFactory()

        dfs = {}
        for resource in self.data_package['resources']:
            resource_url = resource['url']

            dfs[resource['name']] = study._load_tsv(resource_url)

        return Study(experiment_design_data=dfs['experiment_design'],
                     expression_data=dfs['expression'],
                     splicing_data=dfs['splicing'])


def embark(data_package_url):
    study = Embark(data_package_url)
    return study.create_package()