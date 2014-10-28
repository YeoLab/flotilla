"""Test utilities interfacing with external-facing modules, e.g. links to
gene lists"""
import os
import sys
import tempfile
import subprocess

import pandas as pd


def test_link_to_list(genelist_link):
    from flotilla.external import link_to_list

    test_list = link_to_list(genelist_link)

    if genelist_link.startswith("http"):
        sys.stderr.write(
            "WARNING, downloading things from the internet, potential danger "
            "from untrusted sources\n")
        filename = tempfile.NamedTemporaryFile(mode='w+')
        filename.write(subprocess.check_output(
            ["curl", "-k", '--location-trusted', genelist_link]))
        filename.seek(0)
    elif genelist_link.startswith("/"):
        assert os.path.exists(os.path.abspath(genelist_link))
        filename = os.path.abspath(genelist_link)
    true_list = pd.read_table(filename, squeeze=True, header=None).values \
        .tolist()

    assert true_list == test_list
