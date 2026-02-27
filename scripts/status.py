#!/usr/bin/env python3
"""
Dynamic project status for tabular embeddings research.

Displays:
- Background jobs (SAE sweeps, embeddings)
- Worker GPU status
- Git status (commits, dirty state)
- Paper deadline progress
"""
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Paper deadline - NeurIPS 2026
PAPER_DEADLINE = datetime(2026, 5, 20)

# Workers
WORKERS = ["surfer4", "terrax4", "octo4", "firelord4"]

# Key milestones for NeurIPS 2026 submission
MILESTONES = {
    "Universal SAE validated": datetime(2026, 3, 10),      # All 6 TFMs with <10% dead neurons
    "Concept analysis complete": datetime(2026, 3, 31),    # Cross-model alignment done
    "First draft complete": datetime(2026, 4, 21),         # Ready for internal review
    "Submission ready": PAPER_DEADLINE,                    # NeurIPS 2026 deadline
}


def check_background_jobs():
    """Check for running jobs on workers."""
    jobs = []
    for worker in WORKERS:
        try:
            # Check for SAE sweeps
            result = subprocess.run(
                ["ssh", worker, "ps aux | grep -E '(sae_tabarena_sweep|compare_embeddings)' | grep -v grep"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                # Extract process info
                for line in result.stdout.strip().split('\n'):
                    if 'sae_tabarena_sweep' in line:
                        # Extract model name from command
                        if '--model' in line:
                            model = line.split('--model')[1].split()[0]
                            jobs.append(f"SAE:{model}@{worker}")
                    elif 'compare_embeddings' in line:
                        jobs.append(f"Embed@{worker}")
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
    return jobs


def check_worker_status():
    """Check GPU worker health (simplified)."""
    online = []
    offline = []
    for worker in WORKERS:
        try:
            result = subprocess.run(
                ["ssh", worker, "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0 and result.stdout.strip():
                online.append(worker)
            else:
                offline.append(worker)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            offline.append(worker)
    return online, offline


def check_git_status():
    """Check git status."""
    try:
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True
        )
        dirty = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

        # Check commits ahead/behind
        result = subprocess.run(
            ["git", "rev-list", "--left-right", "--count", "origin/main...HEAD"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            behind, ahead = map(int, result.stdout.strip().split())
        else:
            behind, ahead = 0, 0

        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True
        )
        branch = result.stdout.strip() or "detached"

        return {
            "branch": branch,
            "dirty": dirty,
            "ahead": ahead,
            "behind": behind,
        }
    except subprocess.SubprocessError:
        return None


def check_paper_deadline():
    """Check progress toward paper deadline."""
    now = datetime.now()
    days_left = (PAPER_DEADLINE - now).days

    # Check current milestone
    current_milestone = None
    next_milestone = None
    for name, date in sorted(MILESTONES.items(), key=lambda x: x[1]):
        if now < date:
            if next_milestone is None:
                next_milestone = (name, date)
        else:
            current_milestone = (name, date)

    if next_milestone:
        milestone_name, milestone_date = next_milestone
        days_to_milestone = (milestone_date - now).days
        # Simple heuristic: are we on track?
        # Assume linear progress through milestones
        total_days = (PAPER_DEADLINE - datetime(2026, 2, 1)).days
        elapsed_days = (now - datetime(2026, 2, 1)).days
        expected_progress = elapsed_days / total_days
        # You'd need to track actual progress % - this is a placeholder
        actual_progress = 0.3  # TODO: compute from completed tasks

        if actual_progress >= expected_progress:
            status = "✓ on track"
        else:
            days_behind = int((expected_progress - actual_progress) * total_days)
            status = f"⚠ {days_behind}d behind"
    else:
        milestone_name = "All milestones passed!"
        days_to_milestone = days_left
        status = "🎯 final sprint"

    return {
        "days_left": days_left,
        "next_milestone": milestone_name,
        "days_to_milestone": days_to_milestone,
        "status": status,
    }


def format_statusline(compact=False):
    """Format statusline for display."""
    jobs = check_background_jobs()
    online, offline = check_worker_status()
    git = check_git_status()
    paper = check_paper_deadline()

    if compact:
        # Compact: one line for statusline
        parts = []

        # Jobs
        if jobs:
            parts.append(f"🔄 {','.join(jobs)}")
        else:
            parts.append("💤 idle")

        # Workers
        parts.append(f"🖥 {len(online)}/{len(WORKERS)}")

        # Git
        if git:
            if git["dirty"] > 0 or git["ahead"] > 0:
                parts.append(f"📝 {git['branch']}")
            else:
                parts.append(f"✓ {git['branch']}")

        # Paper
        if paper:
            parts.append(f"📅 {paper['days_left']}d | {paper['status']}")

        return " | ".join(parts)
    else:
        # Full status report
        lines = []
        lines.append("=" * 70)
        lines.append("TABULAR EMBEDDINGS RESEARCH - PROJECT STATUS")
        lines.append("=" * 70)

        # Background jobs
        lines.append("\n🔄 BACKGROUND JOBS:")
        if jobs:
            for job in jobs:
                lines.append(f"  • {job}")
        else:
            lines.append("  No active jobs")

        # Workers
        lines.append(f"\n🖥  GPU WORKERS ({len(online)}/{len(WORKERS)} online):")
        for worker in online:
            lines.append(f"  ✓ {worker}")
        for worker in offline:
            lines.append(f"  ✗ {worker}")

        # Git
        if git:
            lines.append(f"\n📝 GIT STATUS:")
            lines.append(f"  Branch: {git['branch']}")
            if git["dirty"] > 0:
                lines.append(f"  ⚠ {git['dirty']} uncommitted changes")
            if git["ahead"] > 0:
                lines.append(f"  ↑ {git['ahead']} commits ahead")
            if git["behind"] > 0:
                lines.append(f"  ↓ {git['behind']} commits behind")
            if git["dirty"] == 0 and git["ahead"] == 0 and git["behind"] == 0:
                lines.append(f"  ✓ Clean and synced")

        # Paper deadline
        if paper:
            lines.append(f"\n📅 PAPER DEADLINE:")
            lines.append(f"  Submission: {PAPER_DEADLINE.strftime('%B %d, %Y')}")
            lines.append(f"  Days remaining: {paper['days_left']}")
            lines.append(f"  Next milestone: {paper['next_milestone']} ({paper['days_to_milestone']}d)")
            lines.append(f"  Status: {paper['status']}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Project status")
    parser.add_argument("--compact", action="store_true", help="Compact one-line format")
    parser.add_argument("--update-claude-md", action="store_true", help="Update CLAUDE.md statusline")
    args = parser.parse_args()

    status = format_statusline(compact=args.compact)
    print(status)

    if args.update_claude_md:
        # Update CLAUDE.md with compact statusline
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        if claude_md.exists():
            content = claude_md.read_text()
            # Replace or add statusline
            if "statusline:" in content:
                # Replace existing
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith("statusline:"):
                        new_lines.append(f"statusline: {status}")
                    else:
                        new_lines.append(line)
                content = '\n'.join(new_lines)
            else:
                # Add at top
                content = f"statusline: {status}\n\n" + content

            claude_md.write_text(content)
            print(f"\n✓ Updated CLAUDE.md statusline", file=sys.stderr)


if __name__ == "__main__":
    main()
