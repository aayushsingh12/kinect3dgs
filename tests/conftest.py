import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("KINECT_HW") == "1":
        return
    skip = pytest.mark.skip(reason="set KINECT_HW=1 to run hardware tests")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)
