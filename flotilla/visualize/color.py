import matplotlib as mpl
import seaborn as sns
import brewer2mpl

sns.set_palette('deep')
deep = sns.color_palette('deep')

dark2 = map(mpl.colors.rgb2hex,
            brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors)
set1 = map(mpl.colors.rgb2hex,
           brewer2mpl.get_map('Set1', 'qualitative', 9).mpl_colors)
red = set1[0]
blue = set1[1]
green = set1[2]
purple = set1[3]
orange = set1[4]
yellow = set1[5]
brown = set1[6]
pink = set1[7]
grey = set1[8]

almost_black = '#262626'

purples = sns.color_palette('Purples', 9)

str_to_color = {'red': red, 'blue': blue, 'green': green, 'purple': purple,
                'orange': orange, 'brown': brown, 'pink': pink, 'grey': grey}
