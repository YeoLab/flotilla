"""
Functions to deal with creation and loading of datapackages
"""

import gzip
import json
import os
import string
import sys
import matplotlib as mpl

import six
from six.moves import urllib

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
        req = urllib.Request(url)
        opener = urllib.build_opener()
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
        req = urllib.Request(url)
        opener = urllib.build_opener()
        opened_url = opener.open(req)
        with open(filename, 'w') as f:
            f.write(opened_url.read())
    return filename


def make_study_datapackage(study_name, metadata,
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
                           gene_ontology=None,
                           supplemental_kws=None,
                           host="https://s3-us-west-2.amazonaws.com/",
                           host_destination='flotilla-projects/'):
    """Example code for making a datapackage for a Study"""
    if ' ' in study_name:
        raise ValueError("Datapackage name cannot have any spaces")
    if set(string.ascii_uppercase) & set(study_name):
        raise ValueError("Datapackage can only contain lowercase letters")

    datapackage_dir = '{}/{}'.format(flotilla_dir, study_name)
    try:
        os.makedirs(datapackage_dir)
    except OSError:
        pass

    supplemental_kws = {} if supplemental_kws is None else supplemental_kws

    datapackage = {'name': study_name, 'title': title, 'sources': sources,
                   'licenses': license, 'datapackage_version': version}

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
                                      splicing_feature_kws),
                 'gene_ontology': (gene_ontology, {})}

    datapackage['resources'] = []
    for resource_name, (data, kws) in resources.items():
        if data is None:
            continue

        datapackage['resources'].append({'name': resource_name})
        resource = datapackage['resources'][-1]

        basename = '{}.csv.gz'.format(resource_name)
        data_filename = '{}/{}'.format(datapackage_dir, basename)
        # if six.PY2:
        #     mode = 'wb'
        # else:
        #     mode = 'wt'
        mode = 'wb' if six.PY2 else 'wt'
        with gzip.open(data_filename, mode) as f:
            data.to_csv(f)

        # if isinstance(data.columns, pd.MultiIndex):
        #     resource['header'] = range(len(data.columns.levels))
        # if isinstance(data.index, pd.MultiIndex):
        #     resource['index_col'] = range(len(data.index.levels))
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
            for key, value in kws.items():
                if key == 'phenotype_to_color':
                    value = dict((k, mpl.colors.rgb2hex(v))
                                 if isinstance(v, tuple) else
                                 (k, v)
                                 for k, v in value.items())
                resource[key] = value

    datapackage['resources'].append({'name': 'supplemental'})
    supplemental = datapackage['resources'][-1]
    supplemental['resources'] = []
    for supplemental_name, data in supplemental_kws.items():
        resource = {}

        basename = '{}.csv.gz'.format(supplemental_name)
        data_filename = '{}/{}'.format(datapackage_dir, basename)
        mode = 'wb' if six.PY2 else 'wt'
        with gzip.open(data_filename, mode) as f:
            data.to_csv(f)

        resource['name'] = supplemental_name
        resource['path'] = basename
        resource['compression'] = 'gzip'
        resource['format'] = 'csv'
        supplemental['resources'].append(resource)

    filename = '{}/datapackage.json'.format(datapackage_dir)
    with open(filename, 'w') as f:
        json.dump(datapackage, f, indent=2)
    sys.stdout.write('Wrote datapackage to {}\n'.format(filename))


def name_to_resource(datapackage, name):
    """Get resource with specified name in the datapackage"""
    for resource in datapackage['resources']:
        if resource['name'] == name:
            return resource
    raise ValueError('No resource named {} in this datapackage'.format(name))
