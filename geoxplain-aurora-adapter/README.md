# GeoXplain Aurora Adapter

| [GeoXplain][geoxplain-link] | **[GeoXplain Aurora Adapter][aurora-link]** | [Documentation][docs-link] | [Live demo][demo-link] | [Paper][paper-link] |
| --- | --- | --- | --- | --- |
| Core toolkit | **Aurora package** | User guide and API | Hosted viewer | Full paper |

<details>
<summary><strong>Cite us</strong></summary>

The GeoXplain preprint is available on arXiv and has been submitted to
vis4climate at IEEE VIS 2026. Until the conference version is available, use the arXiv preprint:

Koprolin, C. W., Trentini, L., Soja, B., El-Assady, M., & Humer, C. (2026).
*GeoXplain: On-the-Fly Visual Explanations for Weather Foundation Models*.
arXiv:2607.05655 [cs.HC]. https://doi.org/10.48550/arXiv.2607.05655

```bibtex
@misc{koprolin2026geoxplainontheflyvisualexplanations,
  title={GeoXplain: On-the-Fly Visual Explanations for Weather Foundation Models},
  author={Clemens Walter Koprolin and Leonardo Trentini and Benedikt Soja and Mennatallah El-Assady and Christina Humer},
  year={2026},
  eprint={2607.05655},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  doi={10.48550/arXiv.2607.05655},
  url={https://arxiv.org/abs/2607.05655},
}
```

</details>

On-the-fly XIA attribution for the Microsoft Aurora weather model, with one Python API that can run locally, through a GPU listener, or through a SLURM-backed listener.

This package is maintained in the `geoxplain-aurora-adapter/` directory of the
[Interpreting Aurora's moisture signals repository][study-repository-link].

Supports adapted versions of Saliency, Integrated Gradients, RISE and Vit-CX.

```python
import geoxplain_aurora_adapter as ax

result = ax.run_saliency(
    target=ax.Target.point(
        var="q",
        level=850,
        lat=46.2,
        lon=8.8,
        timestamp="2024-03-20T00:00:00Z",
    ),
    input=["t", "q", "z"],
    # remote=None                    # local GPU process
    # remote="http://gpu01:8765"     # GPU listener
    # remote="http://localhost:8765" # tunnel to login-node listener
)

result.save("ticino_q850_saliency.xia.npz")
```

`XiaResult` is a self-describing `.xia.npz` bundle that can be passed directly to the GeoXplain Aurora visualization widget.

## Recommended Setup

Install the adapter directly from this repository. From the parent repository
root, run:

```bash
python -m pip install -e ./geoxplain-aurora-adapter
geoxplain-aurora-adapter
```

When already in this directory, use `python -m pip install -e .`.

`geoxplain-aurora-adapter setup` asks for the
preferred deployment mode, writes the config needed,
and prints what to install where.

| Profile | Use when | Install |
|---------|----------|---------|
| `client` | This machine only submits requests to a listener. | `[client]` |
| `local` | Notebooks run directly inside a GPU allocation. | `[gpu]` |
| `gpu-listener` | This machine runs an HTTP listener inside an existing GPU allocation. | `[gpu,server]` |
| `login-node` | A login-node listener submits oneshot jobs or keeps a persistent GPU worker warm. | login node: `[server]`; GPU worker: `[gpu,server,client]` |

Shortcut setup commands:

```bash
geoxplain-aurora-adapter setup --client
geoxplain-aurora-adapter setup --local
geoxplain-aurora-adapter setup --gpu-listener
geoxplain-aurora-adapter setup --login-node
```

The saved mode is only a preference. Listener runs can still override it with
`geoxplain-aurora-adapter listen --mode ...`

More detailed installation notes are in [docs/installation.md](docs/installation.md).

## Start The Listener

For `gpu-listener` or `sbatch` installs, start the listener with:

```bash
geoxplain-aurora-adapter listen
```

CLI flags can override config for a single run:

```bash
geoxplain-aurora-adapter listen --yes \
    --mode sbatch-oneshot \
    --account PROJECT_ACCOUNT \
    --partition GPU_PARTITION \
    --time 00:30:00
```

To discard the saved listener config and rerun setup:

```bash
geoxplain-aurora-adapter setup --reset
```

### Network binding and access

The listener's HTTP API is unauthenticated. It therefore binds `127.0.0.1` (loopback) by default and is meant to be reached over an SSH tunnel:

```bash
ssh -L 8765:localhost:8765 <login-node>   # then remote="http://localhost:8765"
```

Only bind a public address (`--host 0.0.0.0`) on a trusted/firewalled network; the listener prints a warning when you do.

## Deployment Modes

| Mode | Where it runs | When to use |
|------|---------------|-------------|
| local | Notebook process on a GPU node | You already have a GPU allocation. |
| gpu-listener | HTTP listener inside a GPU allocation | Clients can reach the GPU node directly or through a tunnel. |
| sbatch-oneshot | Login-node listener plus one SLURM job per request | Default sbatch-backed mode; no warm worker needed. |
| sbatch-persistent | Login-node listener plus one long-lived GPU worker | Faster repeated calls after model warmup. |

The Python call site stays the same. Set `remote="http://..."` to delegate work to a listener.

## API Sketch

```python
target = ax.Target.box(
    var="q",
    level=850,
    lat=46.25,
    lon=8.75,
    size=(1.5, 2.5),
    timestamp="2020-04-20T12:00:00Z",
)

result = ax.run_saliency(target=target, input=["t", "q", "z"])
result = ax.run_ig(target=target, input=["t", "q", "z"], n_steps=32)
result = ax.run_rise(target=target, input=["t", "q", "z"], n_masks=200)
result = ax.run_vit_cx(target=target, input=["t", "q", "z"], n_clusters=256)
```

Batch timeframes return one multi-frame `XiaResult`:

```python
result = ax.run_saliency(
    target=target,
    input=["t", "q", "z"],
    timeframes=6,
    step_hours=6,
    remote="http://localhost:8765",
)
```

Weather overlays use the same local/remote dispatch path:

```python
overlay = ax.pull_overlay(
    "q",
    "2024-04-20",
    level=850,
    remote="http://localhost:8765",
    name="Specific Humidity 850 hPa",
    unit="kg/kg",
)
```

Omit `dates` to infer them from the explanations run this session. With `overlay_time`, this can be shifted by a fixed amount.

## Result Format

Results are saved as `.xia.npz` archives. Each `XiaResult` contains one or more frames, and each frame carries:

- target metadata
- timestamp
- attribution maps keyed by input variable and vertical layer
- per-frame metadata such as target score or runtime

```python
result.save("case.xia.npz")
restored = ax.XiaResult.load("case.xia.npz")
```

## Transport

Remote execution uses the same FastAPI/msgpack protocol for GPU listeners and SLURM-backed listeners:

| Endpoint | Purpose |
|----------|---------|
| `POST /run` | Submit one target. |
| `POST /run_batch` | Submit multiple timeframes. |
| `GET /jobs/{job_id}` | Poll status, ETA, progress, and log tail. |
| `GET /jobs/{job_id}/result` | Fetch the packed `XiaResult`. |
| `GET /health` | Inspect backend mode, queue depth, and resolved config. |

## Adapting To Other Models

This adapter doubles as a template for other model backends: clone the repository, replace the model, data loading, and target resolution, and swap the method runners, while keeping the dispatch, remote execution, and result serialization unchanged. See [adapt the adapter to another model][adapt-docs-link] for a step-by-step guide.

## More Details

- [Installation and setup](docs/installation.md)
- [Remote execution and retention][remote-docs-link]
- [Weather overlays][overlays-docs-link]

[geoxplain-link]: https://github.com/clemenskoprolin/geoxplain
[aurora-link]: https://github.com/clemenskoprolin/interpreting-auroras-moisture-signals/tree/main/geoxplain-aurora-adapter
[study-repository-link]: https://github.com/clemenskoprolin/interpreting-auroras-moisture-signals
[docs-link]: https://clemenskoprolin.github.io/geoxplain/
[adapt-docs-link]: https://clemenskoprolin.github.io/geoxplain/backends/adapt-another-model/
[remote-docs-link]: https://clemenskoprolin.github.io/geoxplain/guides/remote-execution/#retention
[overlays-docs-link]: https://clemenskoprolin.github.io/geoxplain/guides/overlays/#choose-which-field-to-overlay-overlay_time
[demo-link]: https://ckoprolin.ivia.ch/
[paper-link]: https://arxiv.org/abs/2607.05655
