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
from pathlib import Path
from typing import List, Dict, Optional


def escape_for_template_string(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    return text


def parse_title(markdown_text: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
    return match.group(1).strip() if match else "题解"


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
    """Build sidebar navigation as a week -> day accordion.

    A day with a single solution becomes a direct link (like leetcode/daily).
    A day with multiple solutions expands into a nested accordion.
    """
    lines = []

    overview_class = "nav-link active" if current_slug is None else "nav-link"
    lines.append(f'<a class="{overview_class}" href="./index.html">📌 LeetGPU 题解</a>')
    lines.append('<div class="nav-section-title">题目</div>')

    # Resolve current solution's [week, day] path for expand state
    current_path: List[str] = []
    if current_slug is not None:
        for s in solutions:
            if s["slug"] == current_slug:
                if s["week"]:
                    current_path = [s["week"]]
                    if s["day"]:
                        current_path.append(s["day"])
                break

    # Group: week -> day -> [solutions]
    tree: Dict[str, Dict[str, List[Dict]]] = {}
    for s in solutions:
        w = s["week"] or "未分组"
        d = s["day"] or ""
        tree.setdefault(w, {}).setdefault(d, []).append(s)

    for w in sorted(tree.keys(), key=sort_key_numeric):
        days = tree[w]
        is_week_expanded = bool(current_path and current_path[0] == w)
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
            day_solutions = days[d]
            if not d:
                for s in day_solutions:
                    cls = "nav-link active" if current_slug == s["slug"] else "nav-link"
                    lines.append(f'<a class="{cls}" href="./{s["slug"]}.html">{s["title"]}</a>')
            elif len(day_solutions) == 1:
                s = day_solutions[0]
                cls = "nav-link active" if current_slug == s["slug"] else "nav-link"
                lines.append(f'<a class="{cls}" href="./{s["slug"]}.html">{pretty_name(d)}</a>')
            else:
                is_day_expanded = bool(current_path[:2] == [w, d])
                d_expanded_cls = " is-expanded" if is_day_expanded else ""
                d_aria = "true" if is_day_expanded else "false"
                d_icon = "▼" if is_day_expanded else "▶"
                lines.append(f'<div class="nav-accordion-item level-2{d_expanded_cls}">')
                lines.append('  <div class="nav-accordion-header">')
                lines.append(
                    f'    <span class="nav-link week-link">{pretty_name(d)}</span>'
                    f'<button class="nav-accordion-toggle" aria-label="收起/展开 {pretty_name(d)}" aria-expanded="{d_aria}">{d_icon}</button>'
                )
                lines.append('  </div>')
                lines.append('  <div class="nav-accordion-content">')
                lines.append('    <div class="nav-section">')
                for s in day_solutions:
                    cls = "nav-link active" if current_slug == s["slug"] else "nav-link"
                    lines.append(f'<a class="{cls}" href="./{s["slug"]}.html">{s["title"]}</a>')
                lines.append('    </div>')
                lines.append('  </div>')
                lines.append('</div>')

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
    <link rel="stylesheet" href="../css/style.css?v=2">
    <script src="../js/marked.min.js"></script>
    <link href="../css/prism-tomorrow.min.css" rel="stylesheet">
    <script src="../js/prism.min.js"></script>
    <script src="../js/prism-c.min.js"></script>
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
                <a class="back-link" href="./index.html">← 返回题解列表</a>
            </div>
            <article class="content" id="content"></article>
            <div class="day-nav-bottom"><a class="back-link" href="./index.html">← 返回题解列表</a></div>
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
    <script src="../js/main.js?v=3"></script>
</body>
</html>"""


def build_website(leetgpu_dir: Path, output_dir: Path) -> None:
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
            "week": week,
            "day": day,
            "markdown": markdown_text,
        })

    # Group by (week, day) for overview page
    groups: Dict[tuple, List[Dict]] = {}
    for s in solutions:
        key = (s["week"] or "未分组", s["day"] or "")
        groups.setdefault(key, []).append(s)

    overview_markdown = "# LeetGPU 题解\n\n> CUDA 编程挑战题解，配套每日教程的在线练习。\n\n## 题目列表\n\n"
    for key in sorted(groups.keys(), key=lambda k: (sort_key_numeric(k[0]), sort_key_numeric(k[1]))):
        w, d = key
        heading = pretty_name(w) if not d else f"{pretty_name(w)} · {pretty_name(d)}"
        overview_markdown += f"### {heading}\n\n"
        overview_markdown += '<div class="day-cards">\n'
        for s in groups[key]:
            card_label = pretty_name(d) if d else s["title"]
            overview_markdown += (
                f'<a class="day-card" href="./{s["slug"]}.html">\n'
                f'  <div class="day-card-number">{card_label}</div>\n'
                f'  <div class="day-card-title">{s["title"]}</div>\n'
                f'</a>\n'
            )
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
