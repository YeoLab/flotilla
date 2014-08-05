import pandas.util.testing as pdt

from flotilla.data_model import ExpressionData


class TestExpressionData:
    def test_init(self, example_data):
        #TODO: parameterize and test with dropping outliers
        expression = ExpressionData(example_data.expression)
        pdt.assert_frame_equal(expression.original_data,
                               example_data.expression)

        filtered_data = example_data.expression[
            example_data.expression > expression.expression_thresh]
        pdt.assert_frame_equal(expression.data, filtered_data)
