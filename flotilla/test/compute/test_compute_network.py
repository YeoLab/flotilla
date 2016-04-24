from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest


class TestComputeNetwork:
    @pytest.fixture
    def networker(self):
        from flotilla.compute.network import Networker

        return Networker()

    def test_init(self, networker):
        from flotilla.visualize.color import dark2

        assert networker._default_node_color_mapper('x') == dark2[0]
        assert networker._default_node_size_mapper('x') == 300

    def test_adjacency(self, base_data, networker):
        reduced = base_data.reduce()
        reduced.adjacency = networker.adjacency(reduced.reduced_space)

        # TODO: parameterize this
        n_pcs = 5
        use_pc_1, use_pc_2, use_pc_3, use_pc_4 = True, True, True, True

        data = reduced.reduced_space
        total_pcs = data.shape[1]
        use_cols = np.ones(total_pcs, dtype='bool')
        use_cols[n_pcs:] = False
        use_cols = use_cols * np.array(
            [use_pc_1, use_pc_2, use_pc_3, use_pc_4] + [True, ] * (
                total_pcs - 4))
        selected_cols = data.loc[:, use_cols]
        cov = np.cov(selected_cols)
        nrow, ncol = selected_cols.shape
        adjacency = pd.DataFrame(np.tril(cov * - (np.identity(nrow) - 1)),
                                 index=selected_cols.index, columns=data.index)
        pdt.assert_frame_equal(reduced.adjacency, adjacency)

    def test_graph(self):
        pass
