#!/bin/bash
# Daily run of the Claude skill alpinac-unknowns on the newest not-yet-analysed air run.
# Crontab example:  30 7 * * * /path/to/esper/scripts/run_claude_daily.sh
# Override paths with environment variables if they differ on this machine.
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
SKILL_SRC="$REPO/skills/alpinac-unknowns"
SKILL_LINK="$HOME/.claude/skills/alpinac-unknowns"
TOF_DATA="${TOF_DATA:-/media/Shuttle/TOF-Data}"
SHARED="${SHARED:-/shared}"
export ALPINAC_UNKNOWNS_DIR="${ALPINAC_UNKNOWNS_DIR:-$HOME/.local/share/alpinac/unknowns}"
STATE="$ALPINAC_UNKNOWNS_DIR"
LOGDIR="$STATE/cron_logs"
STAMP="$(date +%Y%m%d_%H%M)"
WORK="$STATE/work/$STAMP"
KEEP_DAYS=60

command -v claude >/dev/null || { echo "claude not found in PATH (install: curl -fsSL https://claude.ai/install.sh | bash)" >&2; exit 1; }

# Install the skill on this machine (symlink to the repo copy); replaces a stale or dangling link,
# but never a real directory someone put there
mkdir -p "$HOME/.claude/skills"
if [ -L "$SKILL_LINK" ] || [ ! -e "$SKILL_LINK" ]; then
  ln -sfn "$SKILL_SRC" "$SKILL_LINK"
elif [ "$(readlink -f "$SKILL_LINK")" != "$SKILL_SRC" ]; then
  echo "warning: $SKILL_LINK is not a link to $SKILL_SRC, using it as is" >&2
fi

# Findings log, created empty on a new machine
mkdir -p "$LOGDIR" "$STATE/reports" "$WORK"
[ -f "$STATE/runs.csv" ] || echo "run,analysed_on,top,n_flagged,n_new_findings,report" > "$STATE/runs.csv"
[ -f "$STATE/findings.md" ] || printf '# alpinac-unknowns: findings log\n\n## Open issues\n\n## Library / sheet issues\n\n## Settled\n\n## Unexplained\n' > "$STATE/findings.md"

# One run at a time; record skipped runs instead of exiting silently
exec 9>"$STATE/.lock"
flock -n 9 || { echo "$(date '+%F %T') previous run still active, skipped" >> "$LOGDIR/skipped.log"; exit 0; }

# Drop old logs and work directories
find "$LOGDIR" -name '*.log' ! -name skipped.log -mtime +"$KEEP_DAYS" -delete
find "$STATE/work" -mindepth 1 -maxdepth 1 -type d -mtime +"$KEEP_DAYS" -exec rm -rf {} +

# Scripts write png/csv to the cwd: run in a fresh work directory, not in the repo.
# PYTHONPATH lets ad-hoc `python -c "import common"` work from there.
cd "$WORK"
export PYTHONPATH="$SKILL_SRC/scripts"

# Unattended: only python (the skill scripts), read-only shell commands, and edits to the findings
# log / reports / the skill's own SKILL.md (Known pitfalls). Note that python itself can run
# anything, so this guards against accidental shell commands, it is not a sandbox.
ALLOWED=(
  "Bash(/shared/pyenv/bin/python:*)"
  "Bash(ls:*)" "Bash(head:*)" "Bash(tail:*)" "Bash(grep:*)" "Bash(wc:*)" "Bash(date:*)"
  "Read" "Glob" "Grep" "Skill"
  "Edit(/$STATE/**)"
  "Edit(/$SKILL_SRC/SKILL.md)" "Edit(/$SKILL_LINK/SKILL.md)"
)

claude -p "Use the alpinac-unknowns skill on the newest not-yet-analysed air run. Follow every step including updating the findings log and saving the report." \
    --allowedTools "${ALLOWED[@]}" \
    --add-dir "$TOF_DATA" "$SHARED" "$HOME/.config/alpinac" "$STATE" "$SKILL_SRC" \
    --max-turns 150 \
  > "$LOGDIR/$STAMP.log" 2>&1
