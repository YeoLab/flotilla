"""

"""

import urllib2
import simplejson

class Embark(object):
    """
    Begin your journey of data exploration
    """

    def __init__(self, datapackage_url):
        self.datapackage_url = datapackage_url
        self.datapackage = self.fetch_datapackage()

    def fetch_datapackage(self):
        req = urllib2.Request(self.datapackage_url)
        opener = urllib2.build_opener()
        f = opener.open(req)
        return simplejson.load(f)

