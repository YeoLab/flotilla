"""

"""

import urllib2

import simplejson

from .data_model.study import StudyFactory, Study


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
        return simplejson.load(f)

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

        return Study(phenotype_data=dfs['experiment_design'],
                     expression_data=dfs['expression'],
                     splicing_data=dfs['splicing'])