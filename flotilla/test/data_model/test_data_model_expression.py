from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import pandas.util.testing as pdt


class TestExpressionData:
    def test_init(self, expression_data_no_na,
                  expression_log_base,
                  expression_plus_one, expression_thresh):
        from flotilla.data_model import ExpressionData

        expression = ExpressionData(expression_data_no_na.copy(),
                                    log_base=expression_log_base,
                                    plus_one=expression_plus_one,
                                    thresh=expression_thresh)
        data = expression_data_no_na.copy()
        thresh = float(expression_thresh)

        if expression_plus_one:
            data += 1
            thresh += 1

        if expression_log_base is not None:
            data = np.divide(np.log(data), np.log(expression_log_base))
            thresh = np.divide(np.log(thresh), np.log(expression_log_base))

        pdt.assert_equal(expression.plus_one, expression_plus_one)
        pdt.assert_equal(expression.log_base, expression_log_base)
        pdt.assert_equal(expression.thresh, thresh)
        pdt.assert_frame_equal(expression.data_original,
                               expression_data_no_na)
        pdt.assert_frame_equal(expression.data, data)
