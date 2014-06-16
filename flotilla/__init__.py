__author__ = 'lovci'



import _cargo_commonObjects as cargo
import _frigate_compute as frigate
import _carrier_DBconnection as carrier
import _barge_utils as barge
import _schooner_data_model as schooner
import _submaraine_viz as submarine


def call_main():
    args = argparser.parse_args()
    params = vars(args)
    main(params)

def main():
    pass