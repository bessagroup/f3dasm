import os
from tempfile import TemporaryDirectory

import pytest

from f3dasm.experiment import FileHandler

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def tmpdir():
    with TemporaryDirectory() as tmp:
        yield tmp


def test_retrieve_all_files(tmpdir):
    os.makedirs(os.path.join(tmpdir, "subdir"), exist_ok=True)
    open(os.path.join(tmpdir, "file1.txt"), "w").close()
    open(os.path.join(tmpdir, "file2.csv"), "w").close()
    open(os.path.join(tmpdir, "file3.txt"), "w").close()
    open(os.path.join(tmpdir, "file4.doc"), "w").close()
    open(os.path.join(tmpdir, "subdir", "file5.txt"), "w").close()
    open(os.path.join(tmpdir, "subdir", "file6.csv"), "w").close()

    fh = FileHandler(tmpdir, "txt")
    result = fh.retrieve_tracked_files()

    assert len(result) == 2
    assert os.path.join(tmpdir, "file1.txt") in result
    assert os.path.join(tmpdir, "file3.txt") in result


def test_retrieve_files_to_process(tmpdir):
    os.makedirs(os.path.join(tmpdir, "subdir"), exist_ok=True)
    open(os.path.join(tmpdir, "file1.txt"), "w").close()
    open(os.path.join(tmpdir, "file2.csv"), "w").close()
    open(os.path.join(tmpdir, "file3.txt"), "w").close()
    open(os.path.join(tmpdir, "file4.doc"), "w").close()
    open(os.path.join(tmpdir, "subdir", "file5.txt"), "w").close()
    open(os.path.join(tmpdir, "subdir", "file6.csv"), "w").close()

    fh = FileHandler(tmpdir, "txt")
    fh.processed_files = [os.path.join(tmpdir, "file1.txt")]
    result = fh.retrieve_files_to_process()

    assert len(result) == 1
    assert os.path.join(tmpdir, "file3.txt") in result


def test_tick_processed(tmpdir):
    os.makedirs(os.path.join(tmpdir, "subdir"), exist_ok=True)
    open(os.path.join(tmpdir, "file1.txt"), "w").close()

    fh = FileHandler(tmpdir, "txt")
    fh.tick_processed(os.path.join(tmpdir, "file1.txt"), 0)

    assert os.path.join(tmpdir, "file1.txt") in fh.processed_files
    assert os.path.join(tmpdir, "file1.txt") not in fh.error_files

    fh.tick_processed(os.path.join(tmpdir, "file1.txt"), 1)

    assert os.path.join(tmpdir, "file1.txt") in fh.processed_files
    assert os.path.join(tmpdir, "file1.txt") in fh.error_files


def test_execute(tmpdir):
    os.makedirs(os.path.join(tmpdir, "subdir"), exist_ok=True)
    open(os.path.join(tmpdir, "file1.txt"), "w").close()

    fh = FileHandler(tmpdir, "txt")
    result = fh.execute(os.path.join(tmpdir, "file1.txt"))

    assert result == 0
