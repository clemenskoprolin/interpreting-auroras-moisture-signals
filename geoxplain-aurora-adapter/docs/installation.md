# Installation and Setup

Use the setup command as a guide and install the adapter from this repository
with the extras needed by each role.

## Setup Command

From the adapter directory in a repository checkout:

```bash
python -m pip install -e .
geoxplain-aurora-adapter setup
```

The setup command:

- asks for the preferred deployment mode;
- writes only the adapter config needed for that mode;
- prints the editable source-install command for each machine/environment

Run a specific setup profile directly:

```bash
geoxplain-aurora-adapter setup --client
geoxplain-aurora-adapter setup --local
geoxplain-aurora-adapter setup --gpu-listener
geoxplain-aurora-adapter setup --login-node
```

`--login-server` is accepted as an alias for `--login-node`.

## Requirements

- Python 3.10 or newer
- `pip`, `uv pip`, Conda, or another site-approved Python installer
- CUDA-capable PyTorch environment for local/GPU-worker compute
- `sbatch` on `PATH` for SLURM listener modes
- WeatherBench2 / ERA5 zarr paths visible from the environment that reads data,
  or network access to the default public Google Cloud WeatherBench2 bucket

## Extras Matrix

| Extras | Use on | Pulls in |
|--------|--------|----------|
| `[client]` | A client machine that only submits requests | `httpx`, `msgpack`; no Torch or FastAPI |
| `[server]` | A host that runs the listener | `fastapi`, `uvicorn`, `msgpack`, `httpx`, plus the zarr/GCS read stack for overlay-on-login; no Torch |
| `[gpu]` | A host that runs Aurora compute | `microsoft-aurora`, `torch`, `xarray`, `pandas`, `zarr`, `gcsfs`, `netCDF4`, image/science helpers |

Common deployments:

| Deployment | Install |
|------------|---------|
| Client-only laptop/login shell | `python -m pip install -e '.[client]'` |
| Notebook directly on a GPU node | `python -m pip install -e '.[gpu]'` |
| GPU listener inside a GPU allocation | `python -m pip install -e '.[gpu,server]'` |
| SLURM login-node listener | `python -m pip install -e '.[server]'` |
| SLURM GPU worker environment | `python -m pip install -e '.[gpu,server,client]'` |

Editable installs from a clone use the same extras:

```bash
python -m pip install -e '.[client]'
python -m pip install -e '.[gpu]'
python -m pip install -e '.[gpu,server]'
python -m pip install -e '.[server]'
```

With `uv pip`, keep the same extras:

```bash
uv pip install -e '.[client]'
uv pip install -e '.[gpu,server]'
uv pip install -e '.[gpu,server,client]'
```

## Profile Notes

### Client

Use this on machines that only submit work to an existing listener.

Required install:

```bash
python -m pip install -e '.[client]'
```

No config is required. Pass the listener URL in API calls:

```python
result = ax.run_saliency(..., remote="http://localhost:8765")
```

### Local GPU Notebook

Use this when the notebook process itself runs inside a GPU allocation.

Required install:

```bash
python -m pip install -e '.[gpu]'
```

Required config:

- `[data].weatherbench2_paths`

Call the API without `remote=`:

```python
result = ax.run_saliency(...)
```

### GPU Listener

Use this when an HTTP listener runs inside an existing GPU allocation.

Required install on the GPU listener environment:

```bash
python -m pip install -e '.[gpu,server]'
```

Required config:

- preferred listener mode: `gpu-listener`
- `[network]` host, port, remote URL
- `[data].weatherbench2_paths`
- `[retention]` memory/result retention windows

Start it with:

```bash
geoxplain-aurora-adapter listen
```

### Login-Node SLURM Listener

Use this when a login-node listener submits GPU work through SLURM.

Required install on the login node:

```bash
python -m pip install -e '.[server]'
```

Required install in the GPU worker environment:

```bash
python -m pip install -e '.[gpu,server,client]'
```

Required config:

- preferred listener mode: `sbatch-oneshot` or `sbatch-persistent`
- `[network]` host, port, remote URL
- `[sbatch]` account, partition, wall time, node/task/GPU shape, worker virtualenv, log/output directories
- optional `[sbatch].extra_srun` for site-specific `srun` options such as container or environment flags
- `[data].weatherbench2_paths`
- `[retention]` memory/result retention windows

Example setup command:

```bash
geoxplain-aurora-adapter setup --login-node \
  --listener-mode sbatch-oneshot \
  --account PROJECT_ACCOUNT \
  --partition GPU_PARTITION \
  --venv ~/venv-aurora-xai
```

On sites that require extra launch options, pass them generically:

```bash
geoxplain-aurora-adapter setup --login-node \
  --extra-srun='--container-image=/path/to/image.sqsh'
```

#### A note for CSCS users

If your GPU worker needs a CSCS environment TOML, pass it through `extra_srun`:

```bash
geoxplain-aurora-adapter setup --login-node \
  --extra-srun='--environment=/path/to/aurora-xai.toml'
```

Start it with:

```bash
geoxplain-aurora-adapter listen
```

`listen --mode ...` can override the preferred listener mode for a single run.
If the saved config does not contain the fields needed for that mode, `listen`
prints the matching setup command, for example:

```bash
geoxplain-aurora-adapter setup --login-node
```

## Config File

Setup writes:

```text
~/.config/geoxplain-aurora-adapter/listen.toml
```

Use `--config <path>` to write or read a different config. CLI flags and
environment variables still override saved values for a single run.

To replace the saved setup:

```bash
geoxplain-aurora-adapter setup --reset
```

`listen` does not create config. If no config exists, it stops and asks you to
run setup first.

To discard listener config before startup:

```bash
geoxplain-aurora-adapter listen --reset
```
