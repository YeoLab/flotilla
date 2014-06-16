from ..data_model import Study


def test_explicit_init(example_data):
    expression, splicing, metadata = example_data
    study = Study(metadata=metadata,
                  expression=expression,
                  splicing=splicing)

def test_args_init(example_data):
    study = Study(*example_data)
