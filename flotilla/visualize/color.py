import matplotlib as mpl
import seaborn as sns
import brewer2mpl

sns.set_palette('deep')
deep = sns.color_palette('deep')

set1 = brewer2mpl.get_map('Set1', 'qualitative', 9).mpl_colors
red = mpl.colors.rgb2hex(set1[0])
blue = mpl.colors.rgb2hex(set1[1])
green = mpl.colors.rgb2hex(set1[2])
purple = mpl.colors.rgb2hex(set1[3])
orange = mpl.colors.rgb2hex(set1[4])
yellow = mpl.colors.rgb2hex(set1[5])
brown = mpl.colors.rgb2hex(set1[6])
pink = mpl.colors.rgb2hex(set1[7])
grey = mpl.colors.rgb2hex(set1[8])

almost_black = '#262626'

purples = sns.color_palette('Purples', 9)

str_to_color = {'red': red, 'blue': blue, 'green': green, 'purple': purple,
                'orange': orange, 'brown': brown, 'pink': pink, 'grey': grey}
