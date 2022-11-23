import os

import torch.distributed as dist


def setup(rank, world_size, backend="gloo"):
    r"""Handles initialising a process group, which needs
    to happen in every process that is run
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend=backend,
        timeout=dist.default_pg_timeout,
        world_size=world_size,
        rank=rank,
    )


def cleanup():
    r"""Destroys all process groups
    """
    dist.destroy_process_group()
