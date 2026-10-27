"""Run the test stub server in its own process for benchmarks: prints the endpoint on the first
line of stdout, then serves until stdin closes or the process is killed.

The stub runs in a separate process so it does not share the client's GIL: in-process, its
request handling would be counted against the library being measured.

    python bench/serve.py [latency_ms] < some-pipe
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))
from stub_server import StubServer  # noqa: E402


def main() -> None:
    latency = float(sys.argv[1]) / 1000 if len(sys.argv) > 1 else 0.0
    with StubServer(latency_s=latency) as server:
        print(server.endpoint, flush=True)
        try:
            sys.stdin.read()  # returns when the parent closes our stdin (or exits)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
