"""The database of a data directory (esper.db)."""

from esper import db



def test_new_data_dir_gitignore(tmp_path):
    """A new data directory gets a .gitignore keeping only its configuration; an
    existing one is left as it is."""
    data_dir = tmp_path / "new"
    db.connect(data_dir, create=True).close()
    gitignore = data_dir / db.GITIGNORE_FILENAME
    assert gitignore.read_text() == db.GITIGNORE
    gitignore.write_text("mine\n")
    db.connect(data_dir, create=True).close()
    assert gitignore.read_text() == "mine\n"
