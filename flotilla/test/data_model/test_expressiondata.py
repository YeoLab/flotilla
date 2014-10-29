import pandas.util.testing as pdt

from flotilla.data_model import ExpressionData


class TestExpressionData:
    def test_init(self, example_data):
        # TODO: parameterize and test with dropping outliers
        expression = ExpressionData(example_data.expression)
        pdt.assert_frame_equal(expression.data,
                               example_data.expression)
