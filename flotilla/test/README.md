# Flotilla Tests

* All files that test something must be named `test_*.py`.
* `conftest.py` gets auto-imported for all tests. If you want something in
`conftest.py` to persist for all testing modules and not get rewritten or
imported, use the decorator `@pytest.fixture(scope="module")`. Otherwise,
if it's just used once use the decorator `@pytest.fixture`.
