__author__ = 'olga'

def test_make_new_package(tmpdir):
    from flotilla.data_model import StudyFactory

    new_study = StudyFactory()
    new_study.sample_metadata = None
    new_study.event_metadata = None
    new_study.expression_metadata = None
    new_study.expression_df = None
    new_study.splicing_df = None
    new_study.event_metadata = None
    new_study.write_package('test_package', 'test_package', install=False,
                            where=tmpdir)

    #then, this will work after a kernel restert
    import test_package  #does nothing, this is an empty package