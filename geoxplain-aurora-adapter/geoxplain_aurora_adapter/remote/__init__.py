"""geoxplain_aurora_adapter.remote — HTTP transport, server backends, and SLURM integration.

This sub-package is only needed on machines that either:
  - Serve requests (GPU or login nodes): install with ``[server]`` extra.
  - Call a remote listener (any machine): install with ``[client]`` extra.

Mac-only users who just load ``.xia.npz`` files via ``geoxplain_aurora_adapter_viz`` do
not need this sub-package at all.
"""
