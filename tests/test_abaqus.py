import pytest
from f3dasm.run_abaqus import main


# @pytest.mark.smoke
def test_abaqus():
    main()


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
