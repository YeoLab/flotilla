"""
Functions to deal with creation and loading of datapackages
"""

import gzip
import json
import os
import string
import sys
import urllib2

import matplotlib as mpl

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


def write_small_or_big_data(data, resource_name, datapackage_dir,
                            max_size=1e7):
    """Save dataframe as a gzipped CSV if small, HDF if large

    "Large" is determined from the product of the data shape, with a maximum
    of `max_size`.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to save
    resource_name : str
        Name of the data for saving
    datapackage_dir : str
        Absolute path, where to save the data to
    max_size : int or float
        Maximum size of the data to save as a "smaller" CSV format, where
        ncol*nrow < max_size

    Returns
    -------
    info : dict
        Information about the written file, e.g. path, format, compression to
        save with the datapackage
    """
    nrow, ncol = data.shape

    info = {}
    if nrow * ncol < max_size:
        # If data is smallish, save as a gzipped csv
        basename = '{}.csv.gz'.format(resource_name)
        data_filename = '{}/{}'.format(datapackage_dir, basename)
        with gzip.open(data_filename, 'wb') as f:
            data.to_csv(f)
            info['compression'] = 'gzip'
        info['format'] = 'csv'
    else:
        # If data is big, save as an HDF file
        basename = '{}.hdf'.format(resource_name)
        data_filename = '{}/{}'.format(datapackage_dir, basename)
        key = 'data'
        info['format'] = 'hdf'
        info['key'] = key
        data.to_hdf(data_filename, key)
    info['path'] = basename
    return info


def make_study_datapackage(study_name, metadata,
                           expression_data=None,
                           splicing_data=None,
                           mapping_stats_data=None,
                           title='',
                           sources='', license=None, species=None,
                           flotilla_dir=FLOTILLA_DOWNLOAD_DIR,
                           metadata_kws=None,
                           expression_kws=None,
                           splicing_kws=None,
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
    if len(study_name.split()) > 1:
        raise ValueError("Datapackage name cannot have any whitespace")
    if set(string.uppercase) & set(study_name):
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

        info = write_small_or_big_data(data, resource_name, datapackage_dir)
        resource.update(info)

        if kws is not None:
            for key, value in kws.iteritems():
                if key == 'phenotype_to_color':
                    value = dict((k, mpl.colors.rgb2hex(v))
                                 if isinstance(v, tuple) else
                                 (k, v)
                                 for k, v in value.iteritems())
                resource[key] = value

    datapackage['resources'].append({'name': 'supplemental'})
    supplemental = datapackage['resources'][-1]
    supplemental['resources'] = []
    for supplemental_name, data in supplemental_kws.items():
        resource = {}
        resource['name'] = supplemental_name

        info = write_small_or_big_data(data, supplemental_name,
                                       datapackage_dir)
        resource.update(info)
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
