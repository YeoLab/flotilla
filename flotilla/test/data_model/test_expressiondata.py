import pandas.util.testing as pdt

from flotilla.data_model import ExpressionData


class TestExpressionData:
    def test_init(self, shalek2013_data):
        # TODO: parameterize and test with dropping outliers
        expression = ExpressionData(shalek2013_data.expression)
        pdt.assert_frame_equal(expression.data,
                               shalek2013_data.expression)
