from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


def test_dark2():
    from flotilla.visualize.color import dark2
    assert(dark2 == ['#1b9e77', '#d95f02', '#7570b3', '#e7298a',
                     '#66a61e', '#e6ab02', '#a6761d', '#666666']
           )


def test_set1():
    from flotilla.visualize.color import set1
    assert(set1 == ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                    '#ffff33', '#a65628', '#f781bf', '#999999']
           )


def test_str_to_color():
    from flotilla.visualize.color import str_to_color
    assert(str_to_color == {'blue': '#377eb8', 'brown': '#a65628',
                            'green': '#4daf4a', 'grey': '#999999',
                            'orange': '#ff7f00', 'pink': '#f781bf',
                            'purple': '#984ea3', 'red': '#e41a1c',
                            'yellow': '#ffff33'}
           )
