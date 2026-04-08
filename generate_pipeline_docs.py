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
