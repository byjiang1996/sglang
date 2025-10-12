import sys
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
if __name__ == "__main__":
    # Simulate CLI arguments (excluding the script name)
    args = [
        "--trust-remote-code",
        "--model-path",
        "Qwen/Qwen3-8B",
        "--attention-backend",
        "triton",
        "--tp",
        "2",
        "--enable-deterministic-inference",
        "--disable-cuda-graph",
        "--skip-server-warmup"
    ]
    server_args = prepare_server_args(args)
    launch_server(server_args)
