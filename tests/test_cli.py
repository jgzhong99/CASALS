from types import SimpleNamespace

from casals_l1b import __main__ as cli


def test_cli_forwards_arguments_to_selected_command(monkeypatch):
    called = {}

    def fake_import(module_name):
        called["module"] = module_name

        def fake_main(argv):
            called["argv"] = argv
            return 0

        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli, "import_module", fake_import)

    assert cli.main(["refh", "export", "--h5", "input.h5"]) == 0
    assert called == {
        "module": "casals_l1b.refh_export",
        "argv": ["--h5", "input.h5"],
    }


def test_cli_help_lists_commands(capsys):
    assert cli.main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "refh" in help_text
    assert "peaks" in help_text
    assert "reference" in help_text
    assert "refh-export" not in help_text
