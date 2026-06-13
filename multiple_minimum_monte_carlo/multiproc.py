"""Module for running functions in parallel on a single node.

This module provides utilities for parallel execution of calculations using
PyTorch multiprocessing. It handles batching of inputs, process management,
and temporary directory cleanup.
"""

import os
import queue as queue_module
import shutil
import time
from typing import List, Dict, Callable, Optional
import torch.multiprocessing as mp
import math
import torch
from pathlib import Path


def batch_dicts(dicts: List[Dict], num_workers: int) -> List[List[Dict]]:
    """
    Batch a list of dictionaries into a list of lists of dictionaries, and add a batch number to each dictionary

    Args:
        dicts (list): list of dictionaries
        num_workers (int): number of workers

    Returns:
        batched_dicts (list): list of lists of dictionaries
    """
    batch_size = math.ceil(len(dicts) / num_workers)
    if batch_size == 0:
        batch_size = 1
    batched_dicts = []
    # Batch the dictionaries and add a batch number to each dictionary
    start_index = 0
    for i in range(num_workers):
        if start_index + batch_size > len(dicts):
            for d in dicts[start_index:]:
                d["batch"] = i
            batched_dicts.append(dicts[start_index:])
        else:
            for d in dicts[start_index : start_index + batch_size]:
                d["batch"] = i
            batched_dicts.append(dicts[start_index : start_index + batch_size])
            start_index += batch_size
            # Recalculate the batch size to ensure that the last batch is not empty
            number_of_items_in_batched_dicts = 0
            for dict in batched_dicts:
                number_of_items_in_batched_dicts += len(dict)
            if i != num_workers - 1:
                batch_size = math.ceil(
                    (len(dicts) - number_of_items_in_batched_dicts)
                    / (num_workers - (i + 1))
                )
    # Remove any empty lists
    # batched_dicts = [x for x in batched_dicts if x != []]

    return batched_dicts


def _get_temp_base_dir() -> Path:
    """Get the best available temporary directory for batch processing.

    Checks for temporary directories in order of preference:
    1. TMPDIR environment variable (commonly set on HPC systems)
    2. /tmp (standard Unix temporary directory)
    3. Current working directory (fallback)

    Returns:
        Path to the best available temporary directory.
    """
    if tmpdir := os.environ.get("TMPDIR"):
        tmp_path = Path(tmpdir)
        if tmp_path.exists() and tmp_path.is_dir():
            return tmp_path

    tmp_path = Path("/tmp")
    if tmp_path.exists() and tmp_path.is_dir():
        return tmp_path

    return Path.cwd()


def run_func(
    func: Callable,
    input_list: List[Dict],
    queue: mp.Queue,
    parallel_batch_folder_location: Optional[Path] = None,
) -> None:
    """
    Run a function in parallel with a list of arguments and puts the results in a queue. Do this in a directory named from the batch number

    Args:
        func (function): function to run
        input_list (list): list of dictionaries with arguments for the function
        queue (mp.Queue): queue to put the results in

    Returns:
        None
    """
    torch.set_num_threads(1)
    # Make and cd into a batch folder to run calculations in
    batch = input_list[0]["batch"]
    original_dir = Path.cwd().resolve()
    temp_base = (
        Path(parallel_batch_folder_location)
        if parallel_batch_folder_location is not None
        else _get_temp_base_dir()
    )

    batch_dir = temp_base / f"mmmc_batch_{batch}_{os.getpid()}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    results = []
    try:
        os.chdir(batch_dir)
        # Run the function on each input dictionary
        # Remove the batch number from the input dictionary
        for input_dict in input_list:
            del input_dict["batch"]

        for input_dict in input_list:
            try:
                result = func(**input_dict)
            except Exception as e:
                print("Error in batch", batch, ":", e)
                continue
            results.append(result)
    finally:
        # Change directory back to the original directory and remove the batch folder
        # Use absolute path to return - avoids stale file handle on ".."
        try:
            os.chdir(original_dir)
        except OSError:
            pass  # Original directory may no longer exist
        shutil.rmtree(batch_dir, ignore_errors=True)

    final_dict = {batch: results}
    queue.put_nowait(final_dict)


def parallel_run_proc(
    func: Callable,
    input_list: List[Dict],
    num_workers: int,
    parallel_batch_folder_location: Optional[Path] = None,
    timeout: Optional[float] = 3600,
) -> List:
    """
    Run a function in parallel with a list of arguments

    Args:
        func: Function to run in parallel.
        input_list: List of argument dictionaries for the function.
        num_workers: Number of parallel worker processes.
        parallel_batch_folder_location: Optional directory for temporary batch folders.
        timeout: Maximum total seconds to wait for all processes. Any processes still
            running after this deadline are terminated. Default is 3600 (1 hour).
            Pass None to wait indefinitely.

    Returns:
        results (list): list of results from the function
    """
    # Batch the input list
    batched_dicts = batch_dicts(input_list, num_workers)

    # Explicitly use the "fork" start method: worker arguments include the
    # calculator object and a (possibly local) callable, which fork inherits via
    # the parent's memory rather than pickling. This is required for torch-based
    # calculators that do not survive "spawn" (the default on macOS / Python 3.8+).
    # Fall back to the platform default where fork is unavailable (e.g. Windows).
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    # Set up the queue and processes
    queue = ctx.Queue()
    num_processes = len(batched_dicts)
    processes = []
    rets = []
    for i in range(num_processes):
        p = ctx.Process(
            target=run_func,
            args=(func, batched_dicts[i], queue, parallel_batch_folder_location),
        )
        p.start()
        processes.append(p)

    deadline = time.monotonic() + timeout if timeout is not None else None

    for _ in processes:
        try:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(
                        "Parallel process timeout reached, terminating remaining processes"
                    )
                    break
                ret = queue.get(timeout=remaining)
            else:
                ret = queue.get()
            rets.append(ret)
        except queue_module.Empty:
            print("Parallel process timeout reached, terminating remaining processes")
            break
        except Exception as e:
            print(f"Error in consumer: {e}")

    for p in processes:
        if p.is_alive():
            p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()

    queue.close()
    # Sort results by batch number and collect
    sorted_rets = sorted(rets, key=lambda d: list(d.keys())[0])
    results = []
    for ret in sorted_rets:
        results.extend(list(ret.values())[0])
    return results
