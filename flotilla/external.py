""" interface with external data sources i.e. GO files, web"""
from __future__ import division

from collections import defaultdict
import gzip
import json
import os
import string
import subprocess
import sys
import urllib2

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


FLOTILLA_DOWNLOAD_DIR = os.path.expanduser('~/flotilla_projects')


def generateOntology(df):
    from collections import defaultdict
    import itertools

    ontology = defaultdict(
        lambda: {'genes': set(), 'name': set(), 'domain': set()})
    allGenesInOntologies = set(df.get('Ensembl Gene ID'))
    for GO, gene, domain, name in itertools.izip(df.get('GO Term Accession'),
                                                 df.get('Ensembl Gene ID'),
                                                 df.get('GO domain'),
                                                 df.get('GO Term Name')):
        ontology[GO]['genes'].add(gene)
        ontology[GO]['name'].add(name)
        ontology[GO]['domain'].add(domain)
        ontology[GO]['n_genes'] = len(ontology[GO]['genes'])
    return ontology, allGenesInOntologies


def GO_enrichment(geneList, ontology, expressedGenes=None, printIt=False,
                  pCut=1000000, xRef={}):
    lenAllGenes, lenTheseGenes = len(expressedGenes), len(geneList)
    pValues = defaultdict()
    nCmps = 0

    for GOTerm, GOGenes in ontology.items():
        inBoth = GOGenes['genes'].intersection(geneList)
        expressedGOGenes = GOGenes['genes'].intersection(expressedGenes)
        if len(inBoth) <= 3 or len(expressedGOGenes) < 5:
            pValues[GOTerm] = 'notest'
            continue
        # survival function is more accurate on small p-values...
        pVal = hypergeom.sf(len(inBoth), lenAllGenes, len(expressedGOGenes),
                            lenTheseGenes)
        if pVal < 0:
            pVal = 0
        symbols = []
        for ensg in inBoth:
            if ensg in xRef:
                symbols.append(xRef[ensg])
            else:
                symbols.append(ensg)
        pValues[GOTerm] = (
            pVal, len(inBoth), len(expressedGOGenes), len(GOGenes['genes']),
            inBoth,
            symbols)

    for k, v in pValues.items():
        try:
            pValues[k][0] = v * float(nCmps)  # bonferroni correction
        except:
            pass
    import operator

    y = []

    sorted_x = sorted(pValues.iteritems(), key=operator.itemgetter(1))

    for k, v in sorted_x:
        if v == "notest":
            continue
        if not type(k) == str:
            continue
        try:
            if v[0] > pCut:
                continue
            if printIt:
                print k, "|".join(ontology[k]['name']), "%.3e" % v[0], v[1], \
                    v[2], v[3], "|".join(v[3])
                pass
            y.append([k, "|".join(ontology[k]['name']), v[0], v[1], v[2], v[3],
                      ",".join(v[4]), ",".join(v[5])])

        except:
            pass

    try:
        df = pd.DataFrame(y, columns=['GO Term ID', 'GO Term Description',
                                      'Bonferroni-corrected Hypergeometric '
                                      'p-Value',
                                      'N Genes in List and GO Category',
                                      'N Expressed Genes in GO Category',
                                      'N Genes in GO category',
                                      'Ensembl Gene IDs in List',
                                      'Gene symbols in List'])
        df.set_index('GO Term ID', inplace=True)
    except:
        df = pd.DataFrame(None, columns=['GO Term ID', 'GO Term Description',
                                         'Bonferroni-corrected Hypergeometric '
                                         'p-Value',
                                         'N Genes in List and GO Category',
                                         'N Expressed Genes in GO Category',
                                         'N Genes in GO category',
                                         'Ensembl Gene IDs in List',
                                         'Gene symbols in List'])

    return df


class GO(object):
    """
    gene ontology tool

    >>> go = hg19GO()
    >>> go.geneXref['ENSG00000100320']
    'RBFOX2'
    >>> data = go.enrichment(list, background)

    """

    def __init__(self, GOFile):
        with gzip.open(GOFile) as file_handle:
            GO_to_ENSG = pd.read_table(file_handle)
        geneXref = defaultdict()
        for k in np.array(
                GO_to_ENSG.get(["Ensembl Gene ID", "Associated Gene Name"])):
            ensg = k[0]
            gene = k[1]
            geneXref[ensg] = gene

        GO, allGenes = generateOntology(GO_to_ENSG)
        self.GO = GO
        self.allGenes = allGenes
        self.geneXref = geneXref

    def enrichment(self, geneList, background=None, **kwargs):
        if background is None:
            background = self.allGenes
        return GO_enrichment(geneList, self.GO, expressedGenes=background,
                             xRef=self.geneXref)

    def geneNames(self, x):
        try:
            return self.geneXref[x]
        except:
            return x

    def link_to_geneNames(self, list_link):
        list = link_to_list(list_link)
        pd.DataFrame(map(self.geneNames, list), index=list)


def link_to_list(link):
    try:
        assert link.startswith("http") or os.path.exists(os.path.abspath(link))
    except:
        raise ValueError("use a link that starts with http or a file path")

    if link.startswith("http"):
        sys.stderr.write(
            "WARNING, downloading things from the internet, potential danger "
            "from untrusted sources\n")
        xx = subprocess.check_output(
            ["curl", "-k", '--location-trusted', link]).split("\n")
    elif link.startswith("/"):
        assert os.path.exists(os.path.abspath(link))
        with open(os.path.abspath(link), 'r') as f:
            xx = map(str.strip, f.readlines())
    return xx


def data_package_url_to_dict(data_package_url):
    filename = check_if_already_downloaded(data_package_url)

    with open(filename) as f:
        data_package = json.load(f)
    return data_package


def check_if_already_downloaded(url, download_dir=FLOTILLA_DOWNLOAD_DIR):
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
    suffix = '/'.join(url.rsplit('/', 2)[1:-1])
    package_dir = '{}/{}'.format(download_dir, suffix)

    try:
        os.mkdir(package_dir)
        sys.stdout.write('Creating a directory for saving the data for this '
                         'project: {}\n'.format(package_dir))
    except OSError:
        pass

    name = url.rsplit('/', 1)[-1]
    filename = os.path.expanduser(os.path.join(package_dir, name))

    if not os.path.isfile(filename):
        sys.stdout.write('{} has not been downloaded before.\n\tDownloading '
                         'now '
                         'to {}\n'.format(url, filename))
        req = urllib2.Request(url)
        opener = urllib2.build_opener()
        opened_url = opener.open(req)
        with open(filename, 'w') as f:
            f.write(opened_url.read())
    return filename


def make_study_datapackage(name, experiment_design_data,
                           expression_data=None, splicing_data=None,
                           spikein_data=None,
                           mapping_stats_data=None,
                           title='',
                           sources='', license=None, species=None,
                           flotilla_dir=FLOTILLA_DOWNLOAD_DIR,
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
    datapackage['license'] = license

    if species is not None:
        datapackage['species'] = species

    resources = {'metadata': experiment_design_data,
                 'expression': expression_data,
                 'splicing': splicing_data,
                 'spikein': spikein_data,
                 'mapping_stats': mapping_stats_data}

    datapackage['resources'] = []
    for resource_name, resource_data in resources.items():
        if resource_data is None:
            continue

        datapackage['resources'].append({'name': resource_name})
        resource = datapackage['resources'][-1]

        data_filename = '{}/{}.csv.gz'.format(datapackage_dir, resource_name)
        with gzip.open(data_filename, 'wb') as f:
            resource_data.to_csv(f)
        try:
            #TODO: only transmit data if it has been updated
            subprocess.call(
                "scp {} {}:{}{}.".format(data_filename, host, host_destination,
                                         name), shell=True)
        except Exception as e:
            sys.stderr.write("error sending data to host: {}".format(e))

        resource['path'] = data_filename
        resource['compression'] = 'gzip'
        resource['format'] = 'csv'

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
