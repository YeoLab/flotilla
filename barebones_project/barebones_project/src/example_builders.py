__author__ = 'lovci'
#not updated/maintained. project-specific. this is an example file
from .utils import data_dir
import gspread
import numpy as np

def build_descriptors():
    raise NotImplementedError
    study_data_dir = data_dir()
    gc = gspread.login('uname', 'pass')

    wks = gc.open('Single-cell_metadata').sheet1

    to_df = []
    for record in wks.get_all_records():
        to_df.append(record)

    descriptors = pd.DataFrame(to_df).set_index("id")

    descriptors['M_cell'] = descriptors['cell_type'] == "M"
    descriptors['neuron_cell'] = np.any([descriptors['cell_type'] == "M", descriptors['cell_type'] == "S"], axis=0)
    descriptors['N_cell'] = descriptors['cell_type'] == "N"
    descriptors['P_cell'] = descriptors['cell_type'] == "P"
    descriptors['S_cell'] = descriptors['cell_type'] == "S"
    descriptors['any_cell'] = True #all cells

    M_color = '#e41a1c'
    N_color = '#4daf4a'
    P_color = '#377eb8'
    S_color = '#FFA500'

    descriptors['cell_color'] = None
    descriptors['cell_color'][descriptors['M_cell']] = M_color
    descriptors['cell_color'][descriptors['N_cell']] = N_color
    descriptors['cell_color'][descriptors['P_cell']] = P_color
    descriptors['cell_color'][descriptors['S_cell']] = S_color

    descriptors['cell_marker'] = None
    descriptors['cell_marker'][descriptors['M_cell']] = 'D'
    descriptors['cell_marker'][descriptors['N_cell']] = 's'
    descriptors['cell_marker'][descriptors['P_cell']] = 'o'
    descriptors['cell_marker'][descriptors['S_cell']] = 'p'
    descriptors['original_name'] = descriptors.index

    descriptors = descriptors.rename_axis(lambda x: descriptors.cell_type[x] + "_" + x, 0)

    descriptors.to_pickle(os.path.join(study_data_dir, "descriptors.df"))
    miso_descriptors = pd.read_pickle(os.path.join(study_data_dir, 'miso_to_ids.df')).set_index('event_name')

    return descriptors, miso_descriptors


def build_transcriptome_data():
    raise NotImplementedError
    descriptors, miso_descriptors = load_descriptors()
    sys.stderr.write("rebuilding psi and rpkm from original files\n")
    #rpkms from ppliu
    rpkms = pd.read_table(original_rpkm_file).set_index("gene_ID")

    # log transform
    rpkms = rpkms.apply(lambda x: np.log10(x+.9))

    #strip gencode version
    rpkms = rpkms.rename_axis(lambda x: x.split(".")[0], 0).T

    #miso psi from obot
    filtered_json_filename = original_splice_file
    miso_psi = pd.read_json(filtered_json_filename)
    miso_psi.set_index(['event_name', 'splice_type'], inplace=True)
    miso_psi = miso_psi.reset_index()

    SEpsis = miso_psi[miso_psi['splice_type'] == "SE"]
    SEpsis = SEpsis.drop(["splice_type", "index"], 1).set_index('event_name')

    rpkms = rpkms.rename_axis(lambda x: descriptors.set_index('original_name')['cell_type'].ix[x] + "_" + x, 0)
    SEpsis = SEpsis.rename_axis(lambda x: descriptors.set_index('original_name')['cell_type'].ix[x] + "_" + x, 1)

    #remove pooled
    not_pooled = set(rpkms.index).difference(set(descriptors.original_name[descriptors['is_pooled'] == 1]))

    #remove non-CVN
    cv_cells = set(rpkms.index).difference(set(descriptors.original_name[descriptors['is_craig?'] == 0]))

    #remove outliers
    not_outliers = set(rpkms.index).difference(set(descriptors.original_name[descriptors['outlier'] == 1]))

    valid_cells = not_pooled & cv_cells & not_outliers

    rpkms = rpkms.ix[valid_cells]
    SEpsis = SEpsis.T.ix[valid_cells]

    rpkms.to_pickle(rpkm_data_dump)
    SEpsis.to_pickle(splice_data_dump)
    return SEpsis, rpkms