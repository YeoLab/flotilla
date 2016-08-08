"""
Convenience functions for obtaining reasonable plotting colors
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import matplotlib as mpl
import seaborn as sns
import brewer2mpl


sns.set_palette('deep')
deep = sns.color_palette('deep')

dark2 = [mpl.colors.rgb2hex(rgb)
         for rgb in brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors]

set1 = [mpl.colors.rgb2hex(rgb)
        for rgb in brewer2mpl.get_map('Set1', 'qualitative', 9).mpl_colors]


[red, blue, green, purple, orange, yellow, brown, pink, grey] = set1

almost_black = '#262626'

purples = sns.color_palette('Purples', 9)

# TODO was yellow omitted before to match 8 only colors in dark2 ?
str_to_color = {'red': red,
                'blue': blue,
                'green': green,
                'purple': purple,
                'orange': orange,
                'yellow': yellow,
                'brown': brown,
                'pink': pink,
                'grey': grey}
