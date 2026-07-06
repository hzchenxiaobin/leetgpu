#!/usr/bin/env python3
"""
Build the LeetGPU solution website from markdown files in leetgpu/.
Generates:
  - index.html: solution list page
  - <slug>.html: individual solution pages
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


def build_nav(current_slug: Optional[str], solutions: List[Dict]) -> str:
    lines = []

    overview_class = "nav-link active" if current_slug is None else "nav-link"
    lines.append(f'<a class="{overview_class}" href="./index.html">📌 LeetGPU 题解</a>')

    lines.append('<div class="nav-section-title">题目</div>')
    for s in solutions:
        cls = "nav-link active" if current_slug == s["slug"] else "nav-link"
        lines.append(f'<a class="{cls}" href="./{s["slug"]}.html">{s["title"]}</a>')

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
    md_files = sorted([
        f for f in leetgpu_dir.iterdir()
        if f.is_file() and f.suffix == ".md" and f.name.startswith("leetgpu-")
    ])

    solutions = []
    for md_file in md_files:
        markdown_text = md_file.read_text(encoding="utf-8")
        # Markdown references images as "images/xxx.svg"; convert to relative path
        markdown_text = markdown_text.replace("](images/", "](./images/")

        title = parse_title(markdown_text)
        slug = md_file.stem  # e.g. leetgpu-vector-add-solution
        solutions.append({
            "slug": slug,
            "title": title,
            "markdown": markdown_text,
        })

    # Build overview page
    cards_html = '<div class="day-cards">\n'
    for s in solutions:
        cards_html += (
            f'<a class="day-card" href="./{s["slug"]}.html">\n'
            f'  <div class="day-card-number">LeetGPU</div>\n'
            f'  <div class="day-card-title">{s["title"]}</div>\n'
            f'</a>\n'
        )
    cards_html += '</div>\n'

    overview_markdown = "# LeetGPU 题解\n\n> CUDA 编程挑战题解，配套每日教程的在线练习。\n\n## 题目列表\n\n" + cards_html

    overview_html = page_template(
        title="LeetGPU 题解",
        nav_html=build_nav(current_slug=None, solutions=solutions),
        markdown=overview_markdown,
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    # Build solution pages
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
