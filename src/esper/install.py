"""Install esper in the desktop: application menu entry, icon and ``esper`` command.

Follows the freedesktop conventions:

- ``<prefix>/share/applications/esper.desktop``
- ``<prefix>/share/icons/hicolor/scalable/apps/esper.svg``
- ``<prefix>/bin/esper`` (link to the executable of this installation)

with prefix ``/usr/local`` for all users (run as root) or ``~/.local`` for the
current user. Optionally a shortcut on the current user's desktop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

APP_ID = "esper"
SYSTEM_PREFIX = Path("/usr/local")
USER_PREFIX = Path.home() / ".local"

DESKTOP_ENTRY = """\
[Desktop Entry]
Version=1.0
Type=Application
Name=esper
GenericName=GC-MS peak browser
Comment=Browse the peaks extracted by alpinac
Exec={exec}
Icon={icon}
Terminal=false
Categories=Science;Chemistry;
Keywords=alpinac;chromatogram;mass spectrum;GC-MS;
StartupWMClass={app_id}
"""


def esper_executable() -> Path:
    """The ``esper`` script of the Python environment running this command."""
    candidate = Path(sys.executable).parent / "esper"
    if candidate.is_file():
        return candidate
    found = shutil.which("esper")
    if found is None:
        raise FileNotFoundError(
            "esper executable not found, is esper installed (pip install)?"
        )
    return Path(found)


def _paths(prefix: Path) -> dict[str, Path]:
    return {
        "desktop": prefix / "share" / "applications" / f"{APP_ID}.desktop",
        "icon": prefix
        / "share"
        / "icons"
        / "hicolor"
        / "scalable"
        / "apps"
        / f"{APP_ID}.svg",
        "bin": prefix / "bin" / "esper",
    }


def _invoking_user() -> tuple[str, Path] | None:
    """(name, home) of the user who ran sudo, if running through sudo."""
    import pwd

    name = os.environ.get("SUDO_USER")
    if not name or not is_root():
        return None
    return name, Path(pwd.getpwnam(name).pw_dir)


def user_desktop_dir() -> Path:
    """Desktop folder of the user (of the one who ran sudo, under sudo)."""
    invoking = _invoking_user()
    command = ["xdg-user-dir", "DESKTOP"]
    if invoking is not None:
        command = ["sudo", "-u", invoking[0], *command]
    try:
        out = subprocess.run(command, capture_output=True, text=True, check=True)
        return Path(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return (invoking[1] if invoking else Path.home()) / "Desktop"


def _run(*command: str) -> None:
    """Run a helper (cache update, ...) if available, ignore failures."""
    if shutil.which(command[0]) is None:
        return
    subprocess.run(command, capture_output=True)


def install(
    system: bool,
    data_dir: Path | None = None,
    desktop_shortcut: bool = False,
    prefix: Path | None = None,
) -> list[Path]:
    """Install the menu entry, the icon and the command; returns the written files."""
    prefix = prefix or (SYSTEM_PREFIX if system else USER_PREFIX)
    paths = _paths(prefix)
    executable = esper_executable()

    command = f"{executable} gui"
    if data_dir is not None:
        command += f' "{Path(data_dir).resolve()}"'

    written = []
    paths["icon"].parent.mkdir(parents=True, exist_ok=True)
    icon = resources.files("esper") / "resources" / f"{APP_ID}.svg"
    paths["icon"].write_bytes(icon.read_bytes())
    written.append(paths["icon"])

    entry = DESKTOP_ENTRY.format(exec=command, icon=APP_ID, app_id=APP_ID)
    paths["desktop"].parent.mkdir(parents=True, exist_ok=True)
    paths["desktop"].write_text(entry)
    paths["desktop"].chmod(0o644)
    written.append(paths["desktop"])

    paths["bin"].parent.mkdir(parents=True, exist_ok=True)
    if paths["bin"].is_symlink() or paths["bin"].exists():
        paths["bin"].unlink()
    paths["bin"].symlink_to(executable)
    written.append(paths["bin"])

    _run("update-desktop-database", str(paths["desktop"].parent))
    _run(
        "gtk-update-icon-cache",
        "--quiet",
        "--ignore-theme-index",
        str(prefix / "share" / "icons" / "hicolor"),
    )

    if desktop_shortcut:
        desktop = user_desktop_dir()
        desktop.mkdir(parents=True, exist_ok=True)
        shortcut = desktop / f"{APP_ID}.desktop"
        # Absolute icon path: works even if the icon theme is not refreshed yet
        shortcut.write_text(
            DESKTOP_ENTRY.format(exec=command, icon=paths["icon"], app_id=APP_ID)
        )
        shortcut.chmod(0o755)
        invoking = _invoking_user()
        if invoking is not None:
            shutil.chown(shortcut, user=invoking[0])
            # GNOME: allow launching without the "untrusted launcher" question
            _run(
                "sudo",
                "-u",
                invoking[0],
                "gio",
                "set",
                str(shortcut),
                "metadata::trusted",
                "true",
            )
        else:
            _run("gio", "set", str(shortcut), "metadata::trusted", "true")
        written.append(shortcut)
    return written


def uninstall(system: bool, prefix: Path | None = None) -> list[Path]:
    """Remove what ``install`` wrote (and the current user's desktop shortcut)."""
    prefix = prefix or (SYSTEM_PREFIX if system else USER_PREFIX)
    removed = []
    candidates = list(_paths(prefix).values()) + [
        user_desktop_dir() / f"{APP_ID}.desktop"
    ]
    for path in candidates:
        if path.is_symlink() or path.exists():
            path.unlink()
            removed.append(path)
    _run("update-desktop-database", str(_paths(prefix)["desktop"].parent))
    return removed


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0
