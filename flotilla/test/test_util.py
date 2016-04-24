"""
Test utilities interfacing with external-facing modules,
e.g. links to gene lists
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


def test_timeout():
    pass


def test_serve_ipython():
    pass


def test_dict_to_str():
    from flotilla.util import dict_to_str
    assert(dict_to_str({'a': 1, 'b': 2}) == 'a:1_b:2')
#
#
# def test_install_development_package():
#     pass
#
#
# def test_memoize():
#     pass
#
#
# def test_cached_property():
#     pass
#
#
# def test_as_numpy():
#     pass
#
#
# def test_natural_sort():
#     pass
#
#
# def test_to_base_file_tuple():
#     pass
#
#
# def test_add_package_data_resource():
#     pass
#
#
# def test_validate_params():
#     pass
#
#
# def test_load_pickle_df():
#     pass
#
#
# def test_write_pickle_df():
#     pass
#
#
# def test_load_gzip_pickle_df():
#     pass
#
#
# def test_write_gzip_pickle_df():
#     pass
#
#
# def test_load_tsv():
#     pass
#
#
# def test_load_json():
#     pass
#
#
# def test_write_tsv():
#     pass
#
#
# def test_load_csv():
#     pass
#
#
# def test_write_csv():
#     pass
#
#
# def test_load_hdf():
#     pass
#
#
# def test_write_hdf():
#     pass
#
#
# def test_get_loading_method():
#     pass
#
#
# def test_timestamp():
#     pass
#
#
# def test_AssertionError():
#     pass


def test_link_to_list():
    pass
    # test_list = link_to_list(genelist_link)
    #
    # if genelist_link.startswith("http"):
    #     sys.stderr.write(
    #         "WARNING, downloading things from the internet, potential"
    #         " danger from untrusted sources\n")
    #     filename = tempfile.NamedTemporaryFile(mode='w+')
    #     filename.write(subprocess.check_output(
    #         ["curl", "-k", '--location-trusted', genelist_link]))
    #     filename.seek(0)
    # elif genelist_link.startswith("/"):
    #     assert os.path.exists(os.path.abspath(genelist_link))
    #     filename = os.path.abspath(genelist_link)
    # true_list = pd.read_table(filename, squeeze=True, header=None).values \
    #     .tolist()
    #
    # assert true_list == test_list
