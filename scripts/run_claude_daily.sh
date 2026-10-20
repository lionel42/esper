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
STATE="${ALPINAC_UNKNOWNS_DIR:-$HOME/.local/share/alpinac/unknowns}"
LOGDIR="$STATE/cron_logs"

command -v claude >/dev/null || { echo "claude not found in PATH (install: curl -fsSL https://claude.ai/install.sh | bash)" >&2; exit 1; }

# Install the skill on this machine (symlink to the repo copy)
mkdir -p "$HOME/.claude/skills"
[ -e "$SKILL_LINK" ] || ln -s "$SKILL_SRC" "$SKILL_LINK"

# Findings log, created empty on a new machine
mkdir -p "$LOGDIR" "$STATE/reports"
[ -f "$STATE/runs.csv" ] || echo "run,analysed_on,top,n_flagged,n_new_findings,report" > "$STATE/runs.csv"
[ -f "$STATE/findings.md" ] || printf '# alpinac-unknowns: findings log\n\n## Open issues\n\n## Library / sheet issues\n\n## Settled\n\n## Unexplained\n' > "$STATE/findings.md"

cd "$REPO"
exec flock -n /tmp/alpinac-unknowns.lock \
  claude -p "Use the alpinac-unknowns skill on the newest not-yet-analysed air run. Follow every step including updating the findings log and saving the report." \
    --allowedTools "Bash Read Write Edit Glob Grep Skill" \
    --add-dir "$TOF_DATA" "$SHARED" "$HOME/.local/share/alpinac" "$HOME/.config/alpinac" "$STATE" \
    --max-turns 150 \
  > "$LOGDIR/$(date +%Y%m%d_%H%M).log" 2>&1
