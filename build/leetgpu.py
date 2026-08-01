"""Builder for the LeetGPU solution website."""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .common import REPO_ROOT, page_template

LEETGPU_DIR = REPO_ROOT

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
DIFFICULTY_LABELS = {
    "easy": "Easy · 简单",
    "medium": "Medium · 中等",
    "hard": "Hard · 困难",
}


def _rewrite_md_links_to_html(markdown_text: str) -> str:
    """Rewrite local LeetGPU solution .md links to flat .html pages.

    Solution pages are emitted flat in the public/ output directory, so any
    link to a `leetgpu-<slug>-solution.md` file (regardless of its source path
    prefix) is flattened to `./leetgpu-<slug>-solution.html`. External/README
    links are left untouched.
    """

    def replace_link(match):
        url = match.group(1)
        if not url.endswith(".md"):
            return match.group(0)
        filename = url.rsplit("/", 1)[-1]
        if filename.startswith("leetgpu-") and filename.endswith("-solution.md"):
            new_url = "./" + filename[:-3] + ".html"
            return f"]({new_url})"
        return match.group(0)

    return re.sub(r"\]\((?!https?://|#)([^)]+)\)", replace_link, markdown_text)


def _parse_title(markdown_text: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
    return match.group(1).strip() if match else "题解"


def _strip_leading_h1(markdown_text: str) -> str:
    """Remove the leading '# Title' heading — the page header already renders it."""
    return re.sub(r"^#\s+[^\n]*\n(\n)?", "", markdown_text, count=1)


def _display_title(title: str) -> str:
    """Strip 'LeetGPU ' prefix and ' 题解...' suffix for cleaner labels."""
    t = title
    if t.startswith("LeetGPU "):
        t = t[len("LeetGPU "):]
    t = re.sub(r"\s*题解\s*[（(].*?[）)]$", "", t)
    t = re.sub(r"\s*题解$", "", t)
    return t.strip()


def _difficulty_sort_key(difficulty: str) -> int:
    return DIFFICULTY_ORDER.get(difficulty, 99)


def _number_from_dirname(name: str) -> int:
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else 0


def _build_nav(current_slug: Optional[str], solutions: List[Dict]) -> str:
    """Build sidebar navigation as a difficulty accordion with inline number tags."""
    lines = []

    overview_class = "nav-link active" if current_slug is None else "nav-link"
    lines.append(f'<a class="{overview_class}" href="./index.html">📌 LeetGPU 题解</a>')
    lines.append('<div class="nav-section-title">题目</div>')

    current_difficulty: Optional[str] = None
    if current_slug is not None:
        for s in solutions:
            if s["slug"] == current_slug:
                current_difficulty = s["difficulty"]
                break

    tree: Dict[str, List[Dict]] = {}
    for s in solutions:
        d = s["difficulty"] or "未分组"
        tree.setdefault(d, []).append(s)

    for d in sorted(tree.keys(), key=_difficulty_sort_key):
        items = sorted(tree[d], key=lambda s: s["number"])
        is_expanded = current_difficulty == d
        expanded_cls = " is-expanded" if is_expanded else ""
        aria_expanded = "true" if is_expanded else "false"
        toggle_icon = "▼" if is_expanded else "▶"
        label = DIFFICULTY_LABELS.get(d, d)

        lines.append(f'<div class="nav-accordion-item level-1{expanded_cls}">')
        lines.append('  <div class="nav-accordion-header">')
        lines.append(
            f'    <span class="nav-link week-link">{label}</span>'
            f'<button class="nav-accordion-toggle" aria-label="收起/展开 {label}" aria-expanded="{aria_expanded}">{toggle_icon}</button>'
        )
        lines.append('  </div>')
        lines.append('  <div class="nav-accordion-content">')
        lines.append('    <div class="nav-section">')

        for s in items:
            cls = "nav-link active" if current_slug == s["slug"] else "nav-link"
            lines.append(
                f'<a class="{cls}" href="./{s["slug"]}.html">'
                f'<span class="nav-day-tag">#{s["number"]}</span>'
                f'{s["display_title"]}'
                f'</a>'
            )

        lines.append('    </div>')
        lines.append('  </div>')
        lines.append('</div>')

    lines.append('<div class="nav-section-title">更多</div>')
    lines.append('<a class="nav-link" href="https://hzchenxiaobin.github.io/ai-infra-notes/index.html">📚 AI Infra 学习笔记</a>')
    lines.append('<a class="nav-link" href="https://hzchenxiaobin.github.io/leetcode/">🧩 LeetCode 题解</a>')
    return "\n".join(lines)


def build(public_dir: Path) -> None:
    """Build the LeetGPU website into public_dir/ (root)."""
    output_dir = public_dir

    images_src = LEETGPU_DIR / "images"
    images_dst = output_dir / "images"
    if images_src.exists():
        shutil.copytree(images_src, images_dst, dirs_exist_ok=True)

    md_files = sorted([
        f for f in LEETGPU_DIR.rglob("leetgpu-*.md")
        if f.is_file()
        and "website" not in f.parts
        and "images" not in f.parts
        and "public" not in f.parts
        and "build" not in f.parts
    ])

    solutions = []
    seen_slugs = {}
    for md_file in md_files:
        markdown_text = md_file.read_text(encoding="utf-8")
        markdown_text = markdown_text.replace("](../../images/", "](./images/")
        markdown_text = markdown_text.replace("](images/", "](./images/")
        markdown_text = _rewrite_md_links_to_html(markdown_text)

        title = _parse_title(markdown_text)
        markdown_text = _strip_leading_h1(markdown_text)
        base_slug = md_file.stem

        rel_parts = md_file.relative_to(LEETGPU_DIR).parts
        difficulty = None
        number = 0
        if len(rel_parts) >= 3 and rel_parts[0] in DIFFICULTY_ORDER:
            difficulty = rel_parts[0]
            number = _number_from_dirname(rel_parts[1])

        slug = base_slug
        if slug in seen_slugs:
            seen_slugs[slug] += 1
            if difficulty:
                slug = f"{difficulty}-{number}-{base_slug}"
            else:
                slug = f"{len(seen_slugs)}-{base_slug}"
        else:
            seen_slugs[slug] = 1

        solutions.append({
            "slug": slug,
            "title": title,
            "display_title": _display_title(title),
            "difficulty": difficulty,
            "number": number,
            "markdown": markdown_text,
        })

    unique_solutions = []
    seen_slugs = set()
    for s in solutions:
        if s["slug"] not in seen_slugs:
            seen_slugs.add(s["slug"])
            unique_solutions.append(s)

    def _challenge_slug(solution_slug: str) -> str:
        prefix = "leetgpu-"
        suffix = "-solution"
        if solution_slug.startswith(prefix) and solution_slug.endswith(suffix):
            return solution_slug[len(prefix):-len(suffix)]
        return solution_slug

    problems_json = json.dumps(
        [{"title": s["display_title"], "slug": _challenge_slug(s["slug"])} for s in unique_solutions],
        ensure_ascii=False,
    )

    overview_markdown = f"""<div class="random-pick">
  <button id="random-pick-btn" class="random-btn" data-problems='{problems_json}'>🎲 随机选一道题练习</button>
</div>
<style>
.random-pick {{
  margin: 1rem 0 1.5rem;
  padding: 1rem;
  background: #1f2937;
  border: 1px solid #374151;
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}}
.random-btn {{
  background: #2563eb;
  color: #fff;
  border: none;
  padding: 0.6rem 1.2rem;
  border-radius: 6px;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.2s;
}}
.random-btn:hover {{ background: #1d4ed8; }}
</style>

"""

    diff_groups: Dict[str, List[Dict]] = {}
    for s in solutions:
        d = s["difficulty"] or "未分组"
        diff_groups.setdefault(d, []).append(s)

    difficulties = sorted(diff_groups.keys(), key=_difficulty_sort_key)

    def render_diff_section(d: str) -> str:
        items = sorted(diff_groups[d], key=lambda s: s["number"])
        section = f'<div class="leetcode-section">\n'
        section += f'  <div class="leetcode-section-title">{DIFFICULTY_LABELS.get(d, d)}</div>\n'
        section += f'  <div class="leetcode-problem-list">\n'
        for s in items:
            section += (
                f'    <a class="leetcode-problem-link" href="./{s["slug"]}.html">'
                f'<span class="leetcode-problem-day">#{s["number"]}</span>'
                f'<span class="leetcode-problem-title">{s["display_title"]}</span>'
                f'</a>\n'
            )
        section += '  </div>\n'
        section += '</div>\n\n'
        return section

    for i in range(0, len(difficulties), 3):
        overview_markdown += '<div class="leetcode-overview-row">\n'
        for j in range(3):
            if i + j < len(difficulties):
                col_cls = "leetcode-col-left" if j == 0 else "leetcode-col-middle" if j == 1 else "leetcode-col-right"
                overview_markdown += f'  <div class="leetcode-col {col_cls}">\n'
                overview_markdown += render_diff_section(difficulties[i + j])
                overview_markdown += '  </div>\n'
        overview_markdown += '</div>\n\n'

    root_prefix = ""
    overview_html = page_template(
        title="LeetGPU 题解",
        nav_html=_build_nav(current_slug=None, solutions=solutions),
        markdown=overview_markdown,
        root_prefix=root_prefix,
        sidebar_title="LeetGPU 题解",
        sidebar_title_style="font-size: 1.5rem; margin-bottom: 0;",
        sidebar_href="./index.html",
        show_back_link=False,
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    for s in solutions:
        html = page_template(
            title=s["title"],
            nav_html=_build_nav(current_slug=s["slug"], solutions=solutions),
            markdown=s["markdown"],
            root_prefix=root_prefix,
            sidebar_title="LeetGPU 题解",
            sidebar_title_style="font-size: 1.5rem; margin-bottom: 0;",
            sidebar_href="./index.html",
            show_back_link=False,
        )
        slug_html = f"{s['slug']}.html"
        (output_dir / slug_html).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / slug_html}")
