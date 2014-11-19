import gzip
import json
import os
import string
import sys
import urllib2

import pandas as pd


FLOTILLA_DOWNLOAD_DIR = os.path.expanduser('~/flotilla_projects')


def datapackage_url_to_dict(datapackage_url):
    filename = check_if_already_downloaded(datapackage_url)

    with open(filename) as f:
        datapackage = json.load(f)
    return datapackage


def check_if_already_downloaded(url,
                                datapackage_name=None,
                                download_dir=FLOTILLA_DOWNLOAD_DIR):
    """If a url filename has already been downloaded, don't download it again.

    Parameters
    ----------
    url : str
        HTTP url of a file you want to downlaod

    Returns
    -------
    filename : str
        Location of the file on your system
    """
    try:
        os.mkdir(download_dir)
        sys.stdout.write('Creating a directory for saving your flotilla '
                         'projects: {}\n'.format(download_dir))
    except OSError:
        pass

    if datapackage_name is None:
        req = urllib2.Request(url)
        opener = urllib2.build_opener()
        opened_url = opener.open(req)
        datapackage = json.loads(opened_url.read())
        datapackage_name = datapackage['name']

    package_dir = '{}/{}'.format(download_dir, datapackage_name)

    try:
        os.mkdir(package_dir)
        sys.stdout.write('Creating a directory for saving the data for this '
                         'project: {}\n'.format(package_dir))
    except OSError:
        pass
    basename = url.rsplit('/', 1)[-1]
    filename = os.path.expanduser(os.path.join(package_dir, basename))

    if not os.path.isfile(filename):
        sys.stdout.write('{} has not been downloaded before.\n\tDownloading '
                         'now to {}\n'.format(url, filename))
        req = urllib2.Request(url)
        opener = urllib2.build_opener()
        opened_url = opener.open(req)
        with open(filename, 'w') as f:
            f.write(opened_url.read())
    return filename


def make_study_datapackage(name, metadata,
                           expression_data=None,
                           splicing_data=None,
                           spikein_data=None,
                           mapping_stats_data=None,
                           title='',
                           sources='', license=None, species=None,
                           flotilla_dir=FLOTILLA_DOWNLOAD_DIR,
                           metadata_kws=None,
                           expression_kws=None,
                           splicing_kws=None,
                           spikein_kws=None,
                           mapping_stats_kws=None,
                           version=None,
                           expression_feature_kws=None,
                           expression_feature_data=None,
                           splicing_feature_data=None,
                           splicing_feature_kws=None,
                           host="sauron.ucsd.edu",
                           host_destination='/zfs/www/flotilla_packages/'):
    """Example code for making a datapackage for a Study
    """
    if ' ' in name:
        raise ValueError("Datapackage name cannot have any spaces")
    if set(string.uppercase) & set(name):
        raise ValueError("Datapackage can only contain lowercase letters")

    datapackage_dir = '{}/{}'.format(flotilla_dir, name)
    try:
        os.makedirs(datapackage_dir)
    except OSError:
        pass

    datapackage = {}
    datapackage['name'] = name
    datapackage['title'] = title
    datapackage['sources'] = sources
    datapackage['licenses'] = license
    datapackage['datapackage_version'] = version

    if species is not None:
        datapackage['species'] = species

    resources = {'metadata': (metadata, metadata_kws),
                 'expression': (expression_data, expression_kws),
                 'splicing': (splicing_data, splicing_kws),
                 'spikein': (spikein_data, spikein_kws),
                 'mapping_stats': (mapping_stats_data, mapping_stats_kws),
                 'expression_feature': (expression_feature_data,
                                        expression_feature_kws),
                 'splicing_feature': (splicing_feature_data,
                                      splicing_feature_kws)}

    datapackage['resources'] = []
    for resource_name, (data, kws) in resources.items():
        if data is None:
            continue

        datapackage['resources'].append({'name': resource_name})
        resource = datapackage['resources'][-1]

        basename = '{}.csv.gz'.format(resource_name)
        data_filename = '{}/{}'.format(datapackage_dir, basename)
        with gzip.open(data_filename, 'wb') as f:
            data.to_csv(f)

        if isinstance(data.columns, pd.MultiIndex):
            resource['header'] = range(len(data.columns.levels))
        # try:
        # # TODO: only transmit data if it has been updated
        # subprocess.call(
        # "scp {} {}:{}{}.".format(data_filename, host, host_destination,
        #                                  name), shell=True)
        # except Exception as e:
        #     sys.stderr.write("error sending data to host: {}".format(e))

        resource['path'] = basename
        resource['compression'] = 'gzip'
        resource['format'] = 'csv'
        if kws is not None:
            for key, value in kws.iteritems():
                resource[key] = value

    filename = '{}/datapackage.json'.format(datapackage_dir)
    with open(filename, 'w') as f:
        json.dump(datapackage, f, indent=2)
    sys.stdout.write('Wrote datapackage to {}'.format(filename))


def make_feature_datapackage():
    hg19 = {'name': 'hg19',
            'title': 'Metadata about genes and splicing events',
            'licences': None,
            'sources': 'Gencode and ENSEMBL genes',
            'datapackage_version': '0.1.0',
            'resources': [
                {
                    'format': 'json',
                    'name': 'expression_feature_data',
                    'url': 'http://sauron.ucsd.edu/flotilla_projects/hg19/'
                           'gencode.v19.annotation.gene.attributes.plus.json'
                },
                {
                    'format': 'csv',
                    'name': 'splicing_feature_data',
                    'url': 'http://sauron.ucsd.edu/flotilla_projects/hg19/'
                           'miso_to_ids.csv'
                },
                {
                    'format': 'json',
                    'name': 'gene_ontology_data',
                    'url': 'http://sauron.ucsd.edu/flotilla_projects/hg19/'
                           'ens_to_go.json'
                }
            ]}


def get_resource_from_name(datapackage, name):
    for resource in datapackage['resources']:
        if resource['name'] == name:
            return resource