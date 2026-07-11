#!/usr/bin/env python3
"""
Build the LeetGPU solution website from markdown files in leetgpu/.
Generates:
  - index.html: solution list page (grouped by week/day)
  - <slug>.html: individual solution pages (flat output)

Solutions are organized under leetgpu/weekN/dayM/ (mirroring the daily
tutorial structure). The sidebar groups them as an accordion:
  week -> day -> solution.
Uses relative paths so the site works when deployed under a repository
path prefix (e.g. https://user.github.io/repo-name/leetgpu/).
"""

import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional


def escape_for_template_string(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    return text


def rewrite_md_links_to_html(markdown_text: str) -> str:
    """Rewrite local .md links to .html for GitHub Pages deployment.

    Solution pages are emitted flat in the leetgpu/ output directory, so a
    cross-link like ../../leetgpu/weekN/dayM/leetgpu-xxx-solution.md becomes
    ./leetgpu-xxx-solution.html.
    """

    def replace_link(match):
        url = match.group(1)
        if not url.endswith(".md"):
            return match.group(0)
        new_url = url[:-3] + ".html"
        if new_url.endswith("README.html"):
            new_url = new_url[: -len("README.html")] + "index.html"
        # ../../leetgpu/weekN/dayM/leetgpu-xxx-solution.md -> ./leetgpu-xxx-solution.html
        new_url = re.sub(
            r"^\.\./\.\./leetgpu/week\d+/day\d+/(leetgpu-.*-solution\.html)$",
            r"./\1",
            new_url,
        )
        return f"]({new_url})"

    return re.sub(r"\]\((?!https?://|#)([^)]+)\)", replace_link, markdown_text)


def parse_title(markdown_text: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
    return match.group(1).strip() if match else "题解"


def display_title(title: str) -> str:
    """Strip 'LeetGPU ' prefix and ' 题解...' suffix for cleaner list/sidebar labels."""
    t = title
    if t.startswith("LeetGPU "):
        t = t[len("LeetGPU "):]
    # Remove trailing " 题解（...）" / " 题解(...)" / " 题解"
    t = re.sub(r"\s*题解\s*[（(].*?[）)]$", "", t)
    t = re.sub(r"\s*题解$", "", t)
    return t.strip()


def pretty_name(name: str) -> str:
    """'day1' -> 'Day 1', 'week2' -> 'Week 2'."""
    m = re.match(r"^([a-zA-Z]+)(\d+)$", name)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)}"
    return name


def sort_key_numeric(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def build_nav(current_slug: Optional[str], solutions: List[Dict]) -> str:
    """Build sidebar navigation as a week accordion with inline day tags.

    Each solution is rendered as a single link with its day label and challenge
    title, matching the LeetCode daily navigation style.
    """
    lines = []

    overview_class = "nav-link active" if current_slug is None else "nav-link"
    lines.append(f'<a class="{overview_class}" href="./index.html">📌 LeetGPU 题解</a>')
    lines.append('<div class="nav-section-title">题目</div>')

    # Resolve current solution's week for expand state
    current_week: Optional[str] = None
    if current_slug is not None:
        for s in solutions:
            if s["slug"] == current_slug:
                current_week = s["week"]
                break

    # Group: week -> day -> [solutions]
    tree: Dict[str, Dict[str, List[Dict]]] = {}
    for s in solutions:
        w = s["week"] or "未分组"
        d = s["day"] or ""
        tree.setdefault(w, {}).setdefault(d, []).append(s)

    for w in sorted(tree.keys(), key=sort_key_numeric):
        days = tree[w]
        is_week_expanded = current_week == w
        expanded_cls = " is-expanded" if is_week_expanded else ""
        aria_expanded = "true" if is_week_expanded else "false"
        toggle_icon = "▼" if is_week_expanded else "▶"

        lines.append(f'<div class="nav-accordion-item level-1{expanded_cls}">')
        lines.append('  <div class="nav-accordion-header">')
        lines.append(
            f'    <span class="nav-link week-link">{pretty_name(w)}</span>'
            f'<button class="nav-accordion-toggle" aria-label="收起/展开 {pretty_name(w)}" aria-expanded="{aria_expanded}">{toggle_icon}</button>'
        )
        lines.append('  </div>')
        lines.append('  <div class="nav-accordion-content">')
        lines.append('    <div class="nav-section">')

        for d in sorted(days.keys(), key=sort_key_numeric):
            day_label = d if d else "未分组"
            for s in days[d]:
                cls = "nav-link active" if current_slug == s["slug"] else "nav-link"
                lines.append(
                    f'<a class="{cls}" href="./{s["slug"]}.html">'
                    f'<span class="nav-day-tag">{day_label}</span>'
                    f'{s["display_title"]}'
                    f'</a>'
                )

        lines.append('    </div>')
        lines.append('  </div>')
        lines.append('</div>')

    lines.append('<div class="nav-section-title">更多</div>')
    lines.append('<a class="nav-link" href="../index.html">← Week 1</a>')
    lines.append('<a class="nav-link" href="../week2/index.html">← Week 2</a>')
    lines.append('<a class="nav-link" href="../leetcode/index.html">🧩 LeetCode 题解</a>')
    return "\n".join(lines)


def page_template(title: str, nav_html: str, markdown: str) -> str:
    escaped_markdown = escape_for_template_string(markdown)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="../css/style.css?v=4">
    <script src="../js/marked.min.js"></script>
    <link href="../css/prism-tomorrow.min.css" rel="stylesheet">
    <script src="../js/prism.min.js"></script>
    <script src="../js/prism-c.min.js"></script>
    <script src="../js/prism-cuda.min.js"></script>
    <script src="../js/prism-bash.min.js"></script>
    <script src="../js/prism-python.min.js"></script>
</head>
<body>
    <button class="menu-toggle" aria-label="Toggle menu">☰</button>

    <div class="site-container">
        <aside class="sidebar">
            <div class="sidebar-header">
                <a href="./index.html" style="text-decoration: none;">
                    <h1 class="sidebar-title">AI Infra 学习笔记</h1>
                    <p class="sidebar-subtitle">LeetGPU 题解</p>
                </a>
            </div>
            <nav class="sidebar-nav">
{nav_html}
            </nav>
        </aside>

        <main class="main-content">
            <div class="page-header">
                <h1 class="page-title">{title}</h1>
            </div>
            <article class="content" id="content"></article>
        </main>
    </div>

    <button class="back-to-top" aria-label="Back to top">↑</button>

    <script>
        const markdown = `{escaped_markdown}`;

        const renderer = new marked.Renderer();
        marked.setOptions({{
            renderer: renderer,
            headerIds: false,
            gfm: true,
            breaks: false,
            sanitize: false
        }});

        try {{
            if (typeof marked === 'undefined') {{
                throw new Error('marked.js failed to load.');
            }}
            document.getElementById('content').innerHTML = marked.parse(markdown);

            if (window.Prism) {{
                Prism.highlightAll();
            }}
        }} catch (err) {{
            document.getElementById('content').innerHTML = '<div style="padding: 20px; color: #ff7b72; background: #2d1515; border-radius: 8px;">' +
                '<h2>⚠️ 页面渲染失败</h2>' +
                '<p>' + err.message + '</p>' +
                '</div>';
            console.error('Markdown render error:', err);
        }}
    </script>
    <script src="../js/main.js?v=5"></script>
</body>
</html>"""


def build_website(leetgpu_dir: Path, output_dir: Path) -> None:
    # Copy shared images from leetgpu/images/ to website/images/ so that
    # local preview of website/*.html resolves ./images/xxx.svg without
    # needing the root build.py to copy them separately.
    images_src = leetgpu_dir / "images"
    images_dst = output_dir / "images"
    if images_src.exists():
        shutil.copytree(images_src, images_dst, dirs_exist_ok=True)

    # Recursively find all leetgpu-*.md under leetgpu/ (excludes website/, images/, SKILL.md)
    md_files = sorted([
        f for f in leetgpu_dir.rglob("leetgpu-*.md")
        if f.is_file()
        and "website" not in f.parts
        and "images" not in f.parts
    ])

    solutions = []
    for md_file in md_files:
        markdown_text = md_file.read_text(encoding="utf-8")
        markdown_text = markdown_text.replace("](images/", "](./images/")
        markdown_text = rewrite_md_links_to_html(markdown_text)

        title = parse_title(markdown_text)
        slug = md_file.stem  # e.g. leetgpu-vector-addition-solution

        rel_parts = md_file.relative_to(leetgpu_dir).parts
        week = None
        day = None
        if len(rel_parts) >= 3 and re.match(r"^week\d+$", rel_parts[0]) and re.match(r"^day\d+$", rel_parts[1]):
            week = rel_parts[0]
            day = rel_parts[1]

        solutions.append({
            "slug": slug,
            "title": title,
            "display_title": display_title(title),
            "week": week,
            "day": day,
            "markdown": markdown_text,
        })

    # Group by (week, day) for overview page
    groups: Dict[tuple, List[Dict]] = {}
    for s in solutions:
        key = (s["week"] or "未分组", s["day"] or "")
        groups.setdefault(key, []).append(s)

    overview_markdown = ""

    # Group by week for a LeetCode-style list overview
    weekly_groups: Dict[str, Dict[str, List[Dict]]] = {}
    for s in solutions:
        w = s["week"] or "未分组"
        d = s["day"] or ""
        weekly_groups.setdefault(w, {}).setdefault(d, []).append(s)

    weeks = sorted(weekly_groups.keys(), key=sort_key_numeric)

    def render_week_section(w: str) -> str:
        section = f'<div class="leetcode-section">\n'
        section += f'  <div class="leetcode-section-title">{pretty_name(w)}</div>\n'
        section += f'  <div class="leetcode-problem-list">\n'
        for d in sorted(weekly_groups[w].keys(), key=sort_key_numeric):
            for s in weekly_groups[w][d]:
                section += (
                    f'    <a class="leetcode-problem-link" href="./{s["slug"]}.html">'
                    f'<span class="leetcode-problem-day">{d}</span>'
                    f'<span class="leetcode-problem-title">{s["display_title"]}</span>'
                    f'</a>\n'
                )
        section += '  </div>\n'
        section += '</div>\n\n'
        return section

    # Three-column layout on wider screens: each row contains up to three weeks
    for i in range(0, len(weeks), 3):
        overview_markdown += '<div class="leetcode-overview-row">\n'
        for j in range(3):
            if i + j < len(weeks):
                col_cls = "leetcode-col-left" if j == 0 else "leetcode-col-middle" if j == 1 else "leetcode-col-right"
                overview_markdown += f'  <div class="leetcode-col {col_cls}">\n'
                overview_markdown += render_week_section(weeks[i + j])
                overview_markdown += '  </div>\n'
        overview_markdown += '</div>\n\n'

    overview_html = page_template(
        title="LeetGPU 题解",
        nav_html=build_nav(current_slug=None, solutions=solutions),
        markdown=overview_markdown,
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    for s in solutions:
        html = page_template(
            title=s["title"],
            nav_html=build_nav(current_slug=s["slug"], solutions=solutions),
            markdown=s["markdown"],
        )
        slug_html = f"{s['slug']}.html"
        (output_dir / slug_html).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / slug_html}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    leetgpu_dir = base_dir.parent
    output_dir = base_dir
    build_website(leetgpu_dir, output_dir)
