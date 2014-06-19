__author__ = 'lovci'

from embark import Embark

# import computation as frigate
# import database as carrier
# import util as barge
# import data_model as schooner
# import visualize as submarine


def call_main():

    # where is this "agrparser" initialized?
    args = argparser.parse_args()
    params = vars(args)
    main(params)

def main():
    pass


def embark(datapackge_url):
    return Embark(datapackge_url)