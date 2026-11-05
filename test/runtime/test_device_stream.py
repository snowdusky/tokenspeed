# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from unittest.mock import Mock, patch

import pytest
import torch

from tokenspeed.runtime.utils.device_stream import is_current_stream_capturing


@pytest.mark.parametrize("kind", ["cpu", "meta"])
def test_host_and_meta_have_no_device_stream_capture(kind):
    assert not is_current_stream_capturing(torch.device(kind))


@pytest.mark.parametrize("kind", ["cuda", "privateuseone"])
@pytest.mark.parametrize("capturing", [False, True])
def test_capture_queries_the_device_runtime(kind, capturing):
    module = Mock(is_current_stream_capturing=Mock(return_value=capturing))
    with patch("torch.get_device_module", return_value=module):
        assert is_current_stream_capturing(torch.device(kind)) is capturing
