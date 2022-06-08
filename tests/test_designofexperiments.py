# -*- coding: utf-8 -*-

# from .context import f3dasm

import pytest
from f3dasm.src.designofexperiments import ContinuousParameter

def test_doe():
    with pytest.raises(ValueError):
        a = ContinuousParameter('test', 0.3, 0.2)

if __name__ == '__main__':
    pytest.main()
