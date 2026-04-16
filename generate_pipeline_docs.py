#!/usr/bin/env python
"""
generate_pipeline_docs.py

Reads dvc.yaml and writes docs/pipeline.md automatically.
ENHANCED: Also creates bi-directional links between stage pages and code documentation.

Run this whenever dvc.yaml changes:

    python generate_pipeline_docs.py

Features:
- Auto-generates pipeline.md and pipeline_details.md from dvc.yaml
- Creates individual stage documentation pages in docs/stages/
- Links each stage to its corresponding code documentation
- Validates that all stage scripts have corresponding code docs
- Reports documentation completeness status
"""

import yaml
from pathlib import Path
from typing import Dict, Optional

DVC_YAML  = Path("dvc.yaml")
OUTPUT_MD = Path("docs/pipeline.md")
DETAILED_MD = Path("docs/pipeline_details.md")
CODE_DOC_DIR = Path("docs/code")


def load_dvc(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ==================== Stage-to-Script Mapping ====================

def get_stage_script(stage_name: str, info: dict) -> Optional[str]:
    """Extract the Python script path from a stage's cmd or deps.
    
    Examples:
        `python -m src.torch.vit.predictions_analyze` → `src/torch/vit/predictions_analyze.py`
        `python -m src.data_single` → `src/data_single.py`
    
    Args:
        stage_name: Name of the stage (for error messages)
        info: Stage info dict from dvc.yaml
        
    Returns:
        Path to Python script, or None if not found
    """
    cmd = info.get('cmd', '')
    
    # Try to parse "python -m module.path" format
    if 'python -m ' in cmd:
        module_path = cmd.split('python -m ')[-1].split()[0]
        # Convert dots to slashes: src.torch.vit.predictions_analyze → src/torch/vit/predictions_analyze.py
        script_path = module_path.replace('.', '/') + '.py'
        return script_path
    
    # Fallback: look for .py file in deps (main script, not utility)
    deps = info.get('deps', [])
    # Look for a .py file that's likely the main script (basename matches stage context)
    for d in deps:
        if d.endswith('.py') and not d.endswith('/__init__.py'):
            # Heuristic: if it's the first or most specific dep, it's likely the main script
            if stage_name.replace('-', '_') in d:
                return d
    
    # If no heuristic match, take any .py dep (not ideal but better than nothing)
    for d in deps:
        if d.endswith('.py') and not d.endswith('/__init__.py'):
            return d
    
    return None


def get_code_doc_name(script_path: str) -> str:
    """Convert script path to code doc filename.
    
    Examples:
        `src/torch/vit/predictions_analyze.py` → `predictions_analyze`
        `src/data_single.py` → `data_single`
    """
    return Path(script_path).stem


def code_doc_exists(script_path: str) -> bool:
    """Check if code doc markdown exists for this script."""
    doc_name = get_code_doc_name(script_path)
    doc_path = CODE_DOC_DIR / f"{doc_name}.md"
    return doc_path.exists()


def insert_code_doc_link(stage_page: Path, code_doc_name: str) -> bool:
    """Insert code documentation link at top of stage page if not already present.
    
    Non-destructive: only adds link if it doesn't already exist.
    
    Args:
        stage_page: Path to stage markdown file
        code_doc_name: Filename of code doc (without .md)
        
    Returns:
        True if link was added, False if already present or skipped
    """
    if not stage_page.exists():
        return False
    
    content = stage_page.read_text()
    
    # Check if link already exists
    if f'../code/{code_doc_name}.md' in content or 'Code Documentation' in content or 'Implementation' in content:
        return False
    
    # Insert link after the heading
    lines = content.split('\n')
    if len(lines) > 1:
        # Insert after first heading (typically line 1)
        lines.insert(1, f"\n**Implementation**: [View code documentation](../code/{code_doc_name}.md)\n")
        stage_page.write_text('\n'.join(lines))
        return True
    
    return False


def parse_stage_page(page: Path) -> Dict[str, object]:
    """Extract cmd, deps, and outs from an existing stage markdown page.

    Returns a dict with keys 'cmd' (str), 'deps' (list[str]), 'outs' (list[str]).
    Any section not found is returned as None / empty list.
    """
    text = page.read_text()
    lines = text.split('\n')

    cmd = None
    deps: list[str] = []
    outs: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Command: grab the content of the bash fenced block that follows
        if line.strip() == '**Command**':
            # skip to ```bash
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                i += 1
            i += 1  # skip the ```bash line
            cmd_lines = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                cmd_lines.append(lines[i])
                i += 1
            cmd = '\n'.join(cmd_lines).strip()
            i += 1  # skip closing ```
            continue

        # Dependencies section
        if line.strip() == '**Dependencies**':
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith('- `') and l.endswith('`'):
                    deps.append(l[3:-1])  # strip "- `" and trailing "`"
                elif l.startswith('**') or l.startswith('##') or l == '':
                    if l.startswith('**') or l.startswith('##'):
                        break
                i += 1
            continue

        # Outputs section
        if line.strip() == '**Outputs**':
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith('- `'):
                    # Strip "- `" and anything after the closing "`" (e.g. _(not cached)_)
                    rest = l[3:]
                    end = rest.find('`')
                    if end != -1:
                        outs.append(rest[:end])
                elif l.startswith('**') or l.startswith('##') or l == '':
                    if l.startswith('**') or l.startswith('##'):
                        break
                i += 1
            continue

        i += 1

    return {'cmd': cmd, 'deps': deps, 'outs': outs}


def extract_dvc_outs(info: dict) -> list[str]:
    """Normalise dvc.yaml outs (str or dict) to a flat list of path strings."""
    result = []
    for o in info.get('outs', []):
        if isinstance(o, str):
            result.append(o)
        elif isinstance(o, dict):
            result.extend(o.keys())
    return result


def check_stage_staleness(name: str, info: dict, page: Path) -> list[str]:
    """Compare dvc.yaml stage definition against what is written in the stage page.

    Returns a list of human-readable warning strings (empty list = in sync).
    """
    if not page.exists():
        return []  # new page — not stale, just missing

    parsed = parse_stage_page(page)
    warnings = []

    # --- cmd ---
    dvc_cmd = info.get('cmd', '').strip()
    page_cmd = (parsed['cmd'] or '').strip()
    if dvc_cmd != page_cmd:
        warnings.append(f"  cmd changed:\n    page : {page_cmd!r}\n    dvc  : {dvc_cmd!r}")

    # --- deps ---
    dvc_deps = sorted(info.get('deps', []))
    page_deps = sorted(parsed['deps'])
    if dvc_deps != page_deps:
        added   = sorted(set(dvc_deps) - set(page_deps))
        removed = sorted(set(page_deps) - set(dvc_deps))
        if added:
            warnings.append(f"  deps added  : {added}")
        if removed:
            warnings.append(f"  deps removed: {removed}")

    # --- outs ---
    dvc_outs  = sorted(extract_dvc_outs(info))
    page_outs = sorted(parsed['outs'])
    if dvc_outs != page_outs:
        added   = sorted(set(dvc_outs) - set(page_outs))
        removed = sorted(set(page_outs) - set(dvc_outs))
        if added:
            warnings.append(f"  outs added  : {added}")
        if removed:
            warnings.append(f"  outs removed: {removed}")

    return warnings


def validate_and_report(stages: dict) -> Dict[str, dict]:
    """Validate that all stages have corresponding code documentation.
    
    Returns a dict with:
        - 'documented': stages with code docs
        - 'missing': stages without code docs
        - 'summary': status message
    """
    documented = {}
    missing = {}
    
    for stage_name, info in stages.items():
        script_path = get_stage_script(stage_name, info)
        if script_path:
            if code_doc_exists(script_path):
                doc_name = get_code_doc_name(script_path)
                documented[stage_name] = {
                    'script': script_path,
                    'doc': f"docs/code/{doc_name}.md",
                    'status': '✓'
                }
            else:
                doc_name = get_code_doc_name(script_path)
                missing[stage_name] = {
                    'script': script_path,
                    'doc': f"docs/code/{doc_name}.md",
                    'status': '✗'
                }
        else:
            missing[stage_name] = {
                'script': '(not found)',
                'doc': '(unknown)',
                'status': '?'
            }
    
    return {
        'documented': documented,
        'missing': missing,
        'total_stages': len(stages),
        'documented_count': len(documented),
        'missing_count': len(missing)
    }


def fix_stage_page(name: str, info: dict, page: Path, code_doc_name: Optional[str]) -> bool:
    """Regenerate the auto-generated header of a stale stage page.

    Rewrites cmd / deps / outs from dvc.yaml while preserving every ``##``
    section the user has written (Notes, Output, etc.).

    Returns True if the file was changed.
    """
    if not page.exists():
        return False

    content = page.read_text()
    lines = content.split('\n')

    # Find where manual content begins: first line that starts with '## '
    manual_start = next(
        (i for i, l in enumerate(lines) if l.startswith('## ')),
        None
    )

    # Build the new auto-generated header (without the trailing ## Notes stub)
    header_lines = generate_stage_page(name, info).split('\n')
    notes_start = next(
        (i for i, l in enumerate(header_lines) if l.startswith('## ')),
        None
    )
    if notes_start is not None:
        header_lines = header_lines[:notes_start]
    new_header = '\n'.join(header_lines).rstrip()

    if manual_start is not None:
        manual_tail = '\n'.join(lines[manual_start:])
        new_content = new_header + '\n\n' + manual_tail
    else:
        # No manual sections at all — keep the Notes stub
        new_content = new_header + '\n\n## Notes\n\n_Add your notes about this stage here._\n'

    if new_content == content:
        return False

    page.write_text(new_content)

    # Re-insert the implementation link if it was lost during regeneration
    if code_doc_name:
        insert_code_doc_link(page, code_doc_name)

    return True


def format_list(items: list[str]) -> str:
    """Format a list of paths as inline code, comma separated."""
    return ", ".join(f"`{i}`" for i in items) if items else "—"

def generate_detailed_stages_md(stages: dict) -> str:
    lines = []
    lines.append("# Detailed Pipeline Stages\n")
    lines.append(
        "_Auto-generated from `dvc.yaml`. "
        "Do not edit this file manually — run `generate_pipeline_docs.py` instead._\n"
    )
    # ── Summary table ────────────────────────────────────────
    lines.append("## Stages\n")
    lines.append("| Stage | Command | Outputs |")
    lines.append("|---|---|---|")

    for name, info in stages.items():
        cmd  = f"`{info.get('cmd', '—')}`"
        outs = info.get("outs", [])
        # outs can be list of str or list of dict (when cache: false)
        out_names = []
        for o in outs:
            if isinstance(o, str):
                out_names.append(o)
            elif isinstance(o, dict):
                out_names.extend(o.keys())
        outs_str = format_list(out_names) if out_names else "—"
        lines.append(f"| `{name}` | {cmd} | {outs_str} |")

    lines.append("")

    # ── Per-stage detail ─────────────────────────────────────
    lines.append("## Stage Details\n")

    for name, info in stages.items():
        lines.append(f"### `{name}`\n")
        lines.append(f"**Command**\n```bash\n{info.get('cmd', '')}\n```\n")

        deps = info.get("deps", [])
        if deps:
            lines.append("**Dependencies**\n")
            for d in deps:
                lines.append(f"- `{d}`")
            lines.append("")

        outs = info.get("outs", [])
        if outs:
            lines.append("**Outputs**\n")
            for o in outs:
                if isinstance(o, str):
                    lines.append(f"- `{o}`")
                elif isinstance(o, dict):
                    for path, meta in o.items():
                        cached = meta.get("cache", True) if meta else True
                        note   = "" if cached else " _(not cached)_"
                        lines.append(f"- `{path}`{note}")
            lines.append("")

        lines.append("---\n")
    return "\n".join(lines)

def generate_markdown(stages: dict) -> str:
    lines = []

    lines.append("# Pipeline\n")
    lines.append(
        "_Auto-generated from `dvc.yaml`. "
        "Do not edit this file manually — run `generate_pipeline_docs.py` instead._\n"
    )
    lines.append("Run the full pipeline with:\n")
    lines.append("```bash\ndvc repro\n```\n")

    # My section
    lines.append("## Stages\n")
    for name, info in stages.items():
        lines.append(f"- [{name}](stages/{name}.md)\n")
    return "\n".join(lines)

def generate_stage_page(name: str, info: dict) -> str:
    lines = []
    lines.append(f"# `{name}`\n")
    lines.append("**Command**")
    lines.append(f"```bash\n{info.get('cmd', '')}\n```\n")
    
    # auto-filled from dvc.yaml
    deps = info.get("deps", [])
    if deps:
        lines.append("**Dependencies**")
        for d in deps:
            lines.append(f"- `{d}`")
        lines.append("")

    outs = info.get("outs", [])
    if outs:
        lines.append("**Outputs**")
        for o in outs:
            if isinstance(o, str):
                lines.append(f"- `{o}`")
            elif isinstance(o, dict):
                for path, meta in o.items():
                    cached = meta.get("cache", True) if meta else True
                    note = "" if cached else " _(not cached)_"
                    lines.append(f"- `{path}`{note}")
        lines.append("")

    # manual section — you fill this in, script never overwrites it
    lines.append("## Notes\n")
    lines.append("_Add your notes about this stage here._\n")

    return "\n".join(lines)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate and validate pipeline documentation.")
    parser.add_argument(
        "--fix", action="store_true",
        help="Auto-fix stale stage pages by regenerating cmd/deps/outs from dvc.yaml "
             "while preserving manually-written ## sections."
    )
    args = parser.parse_args()

    data   = load_dvc(DVC_YAML)
    stages = data.get("stages", {})
    md     = generate_markdown(stages)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(md)
    print(f"Written to {OUTPUT_MD}  ({len(stages)} stages)")

    # write summary of stages
    md     = generate_detailed_stages_md(stages)
    DETAILED_MD.parent.mkdir(parents=True, exist_ok=True)
    DETAILED_MD.write_text(md)
    print(f"Written to {DETAILED_MD}  ({len(stages)} stages)")

    # write per-stage pages — but only if they don't exist yet
    stage_dir = Path("docs/stages")
    stage_dir.mkdir(parents=True, exist_ok=True)

    links_added = 0
    
    for name, info in stages.items():
        page = stage_dir / f"{name}.md"
        
        # Get associated script and code doc
        script_path = get_stage_script(name, info)
        code_doc_name = get_code_doc_name(script_path) if script_path else None
        
        if not page.exists():
            # Create new page
            page.write_text(generate_stage_page(name, info))
            print(f"  Created {page}")
            
            # Add code doc link to new page
            if code_doc_name and insert_code_doc_link(page, code_doc_name):
                links_added += 1
                print(f"    → Linked to code/{code_doc_name}.md")
        else:
            # Page exists: try to add link if not present
            if code_doc_name and insert_code_doc_link(page, code_doc_name):
                links_added += 1
                print(f"  Updated {page} with code doc link")
    
    # Staleness detection
    stale_stages = {}
    for name, info in stages.items():
        page = stage_dir / f"{name}.md"
        warnings = check_stage_staleness(name, info, page)
        if warnings:
            stale_stages[name] = warnings

    if stale_stages:
        print("\n" + "="*60)
        print("STALENESS WARNINGS — stage pages out of sync with dvc.yaml")
        print("="*60)
        for stage_name, warns in stale_stages.items():
            print(f"\n⚠️  {stage_name}  ({stage_dir / (stage_name + '.md')})")
            for w in warns:
                print(w)

        if args.fix:
            print("\n── Fixing stale pages ──")
            for stage_name in stale_stages:
                info = stages[stage_name]
                page = stage_dir / f"{stage_name}.md"
                script_path = get_stage_script(stage_name, info)
                code_doc_name = get_code_doc_name(script_path) if script_path else None
                if fix_stage_page(stage_name, info, page, code_doc_name):
                    print(f"  ✓ Fixed {page}")
                else:
                    print(f"  – {page} unchanged (already in sync?)")
        else:
            print("\nRe-run with --fix to auto-update the stale pages,")
            print("or edit them manually and re-run to confirm they are in sync.")
    else:
        print("\n✓ All existing stage pages are in sync with dvc.yaml")

    # Validation and reporting
    print("\n" + "="*60)
    print("DOCUMENTATION VALIDATION")
    print("="*60)
    
    validation = validate_and_report(stages)
    total = validation['total_stages']
    documented = validation['documented_count']
    missing = validation['missing_count']
    
    print(f"\nTotal stages: {total}")
    print(f"  ✓ With code documentation: {documented}")
    print(f"  ✗ Missing code documentation: {missing}")
    
    if validation['missing']:
        print("\n⚠️  MISSING CODE DOCUMENTATION:")
        for stage_name, info in validation['missing'].items():
            script = info.get('script', 'unknown')
            doc = info.get('doc', '(unknown)')
            print(f"   - {stage_name}")
            print(f"     Script: {script}")
            print(f"     Expected doc: {doc}")
    
    if validation['documented']:
        print("\n✓ DOCUMENTED STAGES:")
        for stage_name, info in validation['documented'].items():
            print(f"   - {stage_name} → {info['doc']}")
    
    print(f"\n{links_added} code documentation link(s) added/updated in stage pages")
    print("="*60)
