import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from geoxplain_aurora_adapter.serving.config import (
    DEFAULT_WEATHERBENCH2_PATHS,
    get_config_section,
    write_config,
)
from geoxplain_aurora_adapter.remote.cli import build_parser
from geoxplain_aurora_adapter.remote.sbatch_config import SbatchConfig
from geoxplain_aurora_adapter.serving.listener_modes import normalize_mode
from geoxplain_aurora_adapter.serving.runtime_config import prepare_listener_settings
from geoxplain_aurora_adapter.serving.setup import main as setup_main


class SetupCliTests(unittest.TestCase):
    def _dir(self, td: str, rel: str) -> Path:
        path = Path(td) / rel
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _venv(self, td: str, rel: str = "venv-aurora-xai") -> Path:
        path = self._dir(td, rel)
        bin_dir = self._dir(td, f"{rel}/bin")
        (bin_dir / "activate").write_text("# test venv\n", encoding="utf-8")
        return path

    def test_setup_mode_normalizes_numbered_menu_choices(self):
        self.assertEqual(normalize_mode("1"), "client")
        self.assertEqual(normalize_mode("2"), "local")
        self.assertEqual(normalize_mode("3"), "gpu-listener")
        self.assertEqual(normalize_mode("4"), "login-node")
        self.assertEqual(normalize_mode("mode-4"), "login-node")

    def test_setup_cli_mode_accepts_numbered_menu_choices(self):
        with tempfile.TemporaryDirectory() as td:
            local_config = Path(td) / "local.toml"
            login_config = Path(td) / "login.toml"

            with redirect_stdout(io.StringIO()):
                setup_main(["--yes", "--config", str(local_config), "--mode", "2"])
                setup_main(["--yes", "--config", str(login_config), "--mode", "4"])

            self.assertEqual(get_config_section("setup", local_config)["deployment"], "local")
            login_setup = get_config_section("setup", login_config)
            self.assertEqual(login_setup["deployment"], "login-node")
            self.assertEqual(login_setup["mode"], "sbatch-oneshot")

    def test_login_node_setup_writes_mode_scoped_config(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            wb2 = self._dir(td, "wb2/a.zarr")
            venv = self._venv(td)
            output_dir = self._dir(td, "results")
            out = io.StringIO()
            with redirect_stdout(out):
                setup_main([
                    "--yes",
                    "--config", str(config_path),
                    "--login-node",
                    "--listener-mode", "sbatch-persistent",
                    "--account", "project42",
                    "--partition", "normal",
                    "--extra-srun=--container-image=aurora.sqsh",
                    "--venv", str(venv),
                    "--output-dir", str(output_dir),
                    "--weatherbench2-path", str(wb2),
                ])

            setup = get_config_section("setup", config_path)
            sbatch = get_config_section("sbatch", config_path)
            self.assertEqual(setup["deployment"], "login-node")
            self.assertEqual(setup["mode"], "sbatch-persistent")
            self.assertEqual(sbatch["account"], "project42")
            self.assertEqual(sbatch["partition"], "normal")
            self.assertNotIn("environment", sbatch)
            self.assertEqual(sbatch["extra_srun"], "--container-image=aurora.sqsh")
            self.assertEqual(sbatch["venv"], str(venv.resolve()))
            self.assertEqual(get_config_section("data", config_path)["weatherbench2_paths"], [str(wb2.resolve())])
            text = out.getvalue()
            self.assertIn("-e '.[server]'", text)
            self.assertIn("-e '.[gpu,server,client]'", text)

    def test_login_node_setup_defaults_are_site_agnostic(self):
        with tempfile.TemporaryDirectory() as td:
            scratch = Path(td) / "scratch"
            scratch.mkdir()
            wb2 = self._dir(td, "wb2/a.zarr")
            config_path = Path(td) / "listen.toml"
            out = io.StringIO()
            with patch.dict(os.environ, {"SCRATCH": str(scratch)}):
                with redirect_stdout(out):
                    setup_main([
                        "--yes",
                        "--config", str(config_path),
                        "--login-node",
                        "--weatherbench2-path", str(wb2),
                    ])

            sbatch = get_config_section("sbatch", config_path)
            self.assertEqual(sbatch["account"], "")
            self.assertEqual(sbatch["partition"], "")
            self.assertEqual(sbatch["time"], "01:00:00")
            self.assertEqual(sbatch["venv"], "")
            self.assertEqual(sbatch["extra_srun"], "")
            self.assertEqual(
                sbatch["output_dir"],
                str(scratch / "geoxplain-aurora-adapter"),
            )
            text = out.getvalue()
            self.assertIn("[sbatch].account", text)
            self.assertIn("[sbatch].partition", text)
            self.assertIn("[sbatch].venv", text)
            self.assertNotIn("a122", text)
            self.assertNotIn("cscs", text.lower())

    def test_setup_prints_install_guide_after_config_and_before_next_steps(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            out = io.StringIO()
            with redirect_stdout(out):
                setup_main([
                    "--yes",
                    "--config", str(config_path),
                    "--local",
                ])

            text = out.getvalue()
            self.assertLess(text.index("Configuration"), text.index("Install What Where"))
            self.assertLess(text.index("Install What Where"), text.index("Next Steps"))
            self.assertIn("If not installed yet, please follow the guide above.", text)
            self.assertNotIn("A plain install is enough to run setup", text)
            self.assertNotIn("never creates environments", text)

    def test_local_setup_writes_only_data_config(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            wb2 = self._dir(td, "wb2/a.zarr")
            with redirect_stdout(io.StringIO()):
                setup_main([
                    "--yes",
                    "--config", str(config_path),
                    "--local",
                    "--weatherbench2-path", str(wb2),
                ])

            self.assertEqual(get_config_section("setup", config_path)["deployment"], "local")
            self.assertEqual(get_config_section("data", config_path)["weatherbench2_paths"], [str(wb2.resolve())])
            self.assertEqual(get_config_section("network", config_path), {})
            self.assertEqual(get_config_section("sbatch", config_path), {})

    def test_local_setup_defaults_to_public_weatherbench2_store(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            with redirect_stdout(io.StringIO()):
                setup_main([
                    "--yes",
                    "--config", str(config_path),
                    "--local",
                ])

            self.assertEqual(
                get_config_section("data", config_path)["weatherbench2_paths"],
                list(DEFAULT_WEATHERBENCH2_PATHS),
            )

    def test_client_setup_has_no_listener_sections(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            with redirect_stdout(io.StringIO()):
                setup_main([
                    "--yes",
                    "--config", str(config_path),
                    "--client",
                ])

            self.assertEqual(get_config_section("setup", config_path)["deployment"], "client")
            self.assertEqual(get_config_section("network", config_path), {})
            self.assertEqual(get_config_section("data", config_path), {})
            self.assertEqual(get_config_section("sbatch", config_path), {})

    def test_setup_dry_run_does_not_write_config(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            with redirect_stdout(io.StringIO()):
                setup_main([
                    "--yes",
                    "--config", str(config_path),
                    "--gpu-listener",
                    "--dry-run",
                ])

            self.assertFalse(config_path.exists())

    def test_listen_rejects_local_profile_without_explicit_listener_mode(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            write_config(config_path, {
                "setup": {"deployment": "local"},
                "data": {"weatherbench2_paths": ["/wb2/a.zarr"]},
            })
            args = build_parser().parse_args([
                "--yes",
                "--config", str(config_path),
            ])

            with self.assertRaises(SystemExit) as exc:
                prepare_listener_settings(args, gpu=True, sbatch=False)

            self.assertIn("Not configured to run a listener", str(exc.exception))
            self.assertIn("setup --gpu-listener", str(exc.exception))

    def test_listen_rejects_sbatch_mode_without_sbatch_config(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            write_config(config_path, {
                "setup": {"deployment": "gpu-listener", "mode": "gpu-listener"},
                "data": {"weatherbench2_paths": ["/wb2/a.zarr"]},
                "network": {"host": "127.0.0.1", "port": 8765, "remote_url": "http://localhost:8765"},
            })
            args = build_parser().parse_args([
                "--yes",
                "--config", str(config_path),
                "--mode", "sbatch-oneshot",
            ])

            with self.assertRaises(SystemExit) as exc:
                prepare_listener_settings(args, gpu=False, sbatch=True)

            self.assertIn("Not configured for listener mode", str(exc.exception))
            self.assertIn("[sbatch].account", str(exc.exception))
            self.assertIn("setup --login-node", str(exc.exception))

    def test_listen_requires_setup_when_config_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            args = build_parser().parse_args([
                "--yes",
                "--config", str(config_path),
                "--mode", "sbatch-persistent",
                "--host", "127.0.0.1",
                "--port", "9000",
                "--remote-url", "http://localhost:9000",
                "--account", "project42",
                "--partition", "debug",
                "--time", "01:00:00",
                "--weatherbench2-path", "/wb2/a.zarr",
                "--weatherbench2-path", "/wb2/b.zarr",
            ])

            with self.assertRaises(SystemExit) as exc:
                prepare_listener_settings(args, gpu=False, sbatch=True)

            self.assertIn("No listener config found", str(exc.exception))
            self.assertIn("geoxplain-aurora-adapter setup", str(exc.exception))

    def test_gpu_listener_setup_does_not_write_sbatch_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            wb2 = self._dir(td, "wb2/a.zarr")
            with redirect_stdout(io.StringIO()):
                setup_main([
                    "--yes",
                    "--config", str(config_path),
                    "--gpu-listener",
                    "--weatherbench2-path", str(wb2),
                ])

            self.assertEqual(get_config_section("setup", config_path)["mode"], "gpu-listener")
            self.assertEqual(get_config_section("data", config_path)["weatherbench2_paths"], [str(wb2.resolve())])
            self.assertEqual(get_config_section("sbatch", config_path), {})

    def test_setup_normalizes_pyxis_environment_paths(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            env_dir = home / "aurora-experiments"
            env_dir.mkdir(parents=True)
            env_file = env_dir / "aurora-xai.toml"
            env_file.write_text('image = "/container.sqsh"\n', encoding="utf-8")
            venv = home / "venv-aurora-xai"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "activate").write_text("# test venv\n", encoding="utf-8")
            wb2 = self._dir(td, "wb2/a.zarr")
            output_dir = self._dir(td, "results")
            config_path = Path(td) / "listen.toml"

            with patch.dict(os.environ, {"HOME": str(home)}):
                with redirect_stdout(io.StringIO()):
                    setup_main([
                        "--yes",
                        "--config", str(config_path),
                        "--login-node",
                        "--account", "project42",
                        "--partition", "normal",
                        "--venv", "~/venv-aurora-xai",
                        "--output-dir", str(output_dir),
                        "--extra-srun=--environment=~/aurora-experiments/aurora-xai.toml",
                        "--weatherbench2-path", str(wb2),
                    ])

            sbatch = get_config_section("sbatch", config_path)
            self.assertEqual(sbatch["venv"], str(venv.resolve()))
            self.assertEqual(
                sbatch["extra_srun"],
                f"--environment={env_file.resolve()}",
            )

    def test_setup_rejects_missing_local_paths(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            with self.assertRaises(SystemExit) as exc:
                with redirect_stdout(io.StringIO()):
                    setup_main([
                        "--yes",
                        "--config", str(config_path),
                        "--login-node",
                        "--account", "project42",
                        "--partition", "normal",
                        "--venv", str(Path(td) / "missing-venv"),
                        "--weatherbench2-path", "gs://weatherbench2/datasets",
                    ])

            self.assertIn("[sbatch].venv", str(exc.exception))
            self.assertIn("existing directory", str(exc.exception))

    def test_sbatch_config_normalizes_existing_pyxis_environment_path(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            env_dir = home / "aurora-experiments"
            env_dir.mkdir(parents=True)
            env_file = env_dir / "aurora-xai.toml"
            env_file.write_text('image = "/container.sqsh"\n', encoding="utf-8")
            config_path = Path(td) / "listen.toml"
            write_config(config_path, {
                "sbatch": {
                    "extra_srun": "--environment=~/aurora-experiments/aurora-xai.toml",
                },
            })

            with patch.dict(os.environ, {
                "HOME": str(home),
                "GEOXPLAIN_AURORA_ADAPTER_CONFIG": str(config_path),
            }):
                resolved = SbatchConfig().resolve()

            self.assertEqual(
                resolved.extra_srun,
                f"--environment={env_file.resolve()}",
            )

    def test_listener_existing_config_starts_without_rewrite(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            write_config(config_path, {
                "setup": {"mode": "sbatch-oneshot"},
                "network": {
                    "host": "127.0.0.1",
                    "port": 9001,
                    "remote_url": "http://localhost:9001",
                },
                "sbatch": {
                    "account": "project42",
                    "partition": "debug",
                    "time": "00:45:00",
                    "venv": "~/venv-aurora-xai",
                    "extra_srun": "--container-image=aurora.sqsh",
                },
                "data": {"weatherbench2_paths": ["/wb2/a.zarr"]},
            })

            args = build_parser().parse_args(["--yes", "--config", str(config_path)])
            before = config_path.read_text(encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                settings = prepare_listener_settings(args, gpu=False, sbatch=True)

            after = config_path.read_text(encoding="utf-8")
            self.assertFalse(settings.wrote_config)
            self.assertEqual(settings.mode, "sbatch-oneshot")
            self.assertEqual(settings.port, 9001)
            self.assertEqual(before, after)
            with patch.dict(os.environ, {"GEOXPLAIN_AURORA_ADAPTER_CONFIG": str(config_path)}):
                resolved = SbatchConfig().resolve()
            self.assertEqual(resolved.account, "project42")
            self.assertEqual(resolved.partition, "debug")
            self.assertEqual(resolved.time, "00:45:00")
            self.assertEqual(resolved.extra_srun, "--container-image=aurora.sqsh")

    def test_sbatch_submit_script_uses_only_generic_extra_srun(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = SbatchConfig(
                account="project42",
                partition="gpu",
                venv="~/venv-aurora-xai",
                log_dir=td,
                extra_srun="--container-image=aurora.sqsh",
            ).resolve()

            script = cfg.render_submit_script("python -m geoxplain_aurora_adapter.remote.worker_oneshot")

        self.assertIn("srun -ul --container-image=aurora.sqsh bash -c", script)
        self.assertNotIn("--environment=", script)

    def test_listener_reset_deletes_config_and_requires_setup(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "listen.toml"
            write_config(config_path, {
                "setup": {"mode": "sbatch-oneshot"},
                "network": {
                    "host": "127.0.0.1",
                    "port": 9001,
                    "remote_url": "http://localhost:9001",
                },
                "sbatch": {
                    "account": "old",
                    "partition": "debug",
                    "time": "00:45:00",
                },
                "data": {"weatherbench2_paths": ["/old.zarr"]},
            })
            args = build_parser().parse_args([
                "--yes",
                "--reset",
                "--config", str(config_path),
                "--mode", "sbatch-persistent",
                "--account", "new-account",
                "--partition", "normal",
                "--time", "02:00:00",
                "--host", "0.0.0.0",
                "--port", "8765",
                "--remote-url", "http://localhost:8765",
                "--weatherbench2-path", "/new.zarr",
            ])

            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    prepare_listener_settings(args, gpu=False, sbatch=True)

            self.assertFalse(config_path.exists())
            self.assertIn("No listener config found", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
