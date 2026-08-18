import time
import typing

def format_result(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    inference_time_ms: float,
    runtime_name: str,
    error_msg: typing.Optional[str] = None,
    **kwargs
) -> dict:
    """
    Standardize the result format across different UVDoc implementations.
    """
    result = {
        "input_path": input_path,
        "output_path": output_path,
        "width": width,
        "height": height,
        "inference_time_ms": inference_time_ms,
        "runtime_name": runtime_name,
        "error": error_msg,
    }
    result.update(kwargs)
    return result

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed_ms = (self.end - self.start) * 1000
