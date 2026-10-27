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

"""Build the retained V4.1 Ascend C providers against the active CANN toolkit."""

import json
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent
    package = source.parents[1]
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    compiler = cann / "bin/bisheng"
    includes = [f"-I{cann / 'include'}", f"-I{cann / 'include/ascendc/highlevel_api'}"]
    with tempfile.TemporaryDirectory(prefix="v41-native-build-") as temporary:
        tiling = Path(temporary) / "tiling.o"
        subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-fPIC",
                "-c",
                str(source / "tiling.cpp"),
                "-o",
                str(tiling),
                *includes,
            ],
            check=True,
        )
        for item in json.loads((source / "build.json").read_text()):
            output = package / item["output"]
            command = [
                str(compiler),
                "-xasc",
                "--npu-arch=dav-2201",
                "-O2",
                "-std=c++17",
                "-fPIC",
                "-shared",
                str(source / item["source"]),
            ]
            if item["source"] == "grouped_w4a4.asc":
                command += [
                    "-x",
                    "none",
                    str(tiling),
                    "-ltiling_api",
                    "-lplatform",
                    "-lregister",
                    "-lgraph",
                    "-ldl",
                ]
            output.parent.mkdir(parents=True, exist_ok=True)
            command += [f"-D{define}" for define in item.get("defines", [])]
            command += [
                "-o",
                str(output),
                *includes,
                f"-L{cann / 'lib64'}",
                "-lruntime",
                "-lascendcl",
            ]
            subprocess.run(command, check=True)
            print(f"BUILT {output.name}", flush=True)


if __name__ == "__main__":
    main()
