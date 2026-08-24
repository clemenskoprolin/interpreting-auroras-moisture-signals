import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from geoxplain_aurora_adapter import cli


class UnifiedCliTests(unittest.TestCase):
    def test_setup_subcommand_dispatches_remaining_args(self):
        with patch("geoxplain_aurora_adapter.serving.setup.main") as setup_main:
            cli.main(["setup", "--dry-run", "--client"])

        setup_main.assert_called_once_with(["--dry-run", "--client"])

    def test_no_args_runs_setup_when_config_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "listen.toml"
            with patch("geoxplain_aurora_adapter.cli.resolve_config_path", return_value=missing), \
                    patch("geoxplain_aurora_adapter.serving.setup.main") as setup_main:
                cli.main([])

        setup_main.assert_called_once_with([])

    def test_listen_subcommand_dispatches_remaining_args(self):
        with patch("geoxplain_aurora_adapter.remote.cli.main") as listen_main:
            cli.main(["listen", "--yes", "--mode", "sbatch-oneshot"])

        listen_main.assert_called_once_with(["--yes", "--mode", "sbatch-oneshot"])


if __name__ == "__main__":
    unittest.main()
