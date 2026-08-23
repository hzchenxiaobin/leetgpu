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


# leetgpu.com challenge slugs that differ from the local solution file slug
CHALLENGE_SLUG_ALIASES = {
    "general-matrix-multiplication-gemm": "gemm",
    "sigmoid-activation": "sigmoid",
}


def _rewrite_challenge_links(markdown_text: str, solution_slugs: set) -> str:
    """Rewrite leetgpu.com challenge links inside the 同类练习题 section to the
    local solution page (./leetgpu-<slug>-solution.html) when a solution exists.
    Links outside that section (e.g. the problem statement URL) stay external.
    """

    def replace_link(match):
        slug = match.group(1)
        slug = CHALLENGE_SLUG_ALIASES.get(slug, slug)
        if slug in solution_slugs:
            return f"](./leetgpu-{slug}-solution.html)"
        return match.group(0)

    section = re.search(r"^## 同类练习题\n.*?(?=^## |\Z)", markdown_text, re.MULTILINE | re.DOTALL)
    if not section:
        return markdown_text
    rewritten = re.sub(
        r"\]\(https://leetgpu\.com/challenges/([a-z0-9-]+)\)",
        replace_link,
        section.group(0),
    )
    return markdown_text[: section.start()] + rewritten + markdown_text[section.end() :]


def _rewrite_all_challenge_links(markdown_text: str, solution_slugs: set) -> str:
    """Rewrite all leetgpu.com challenge links to local solution pages when a solution exists.

    Unlike _rewrite_challenge_links (which only rewrites inside the 同类练习题
    section), this rewrites every leetgpu.com/challenges/<slug> link in the
    document — used for topic pages whose challenge references span multiple
    tables/sections.
    """

    def replace_link(match):
        slug = match.group(1)
        slug = CHALLENGE_SLUG_ALIASES.get(slug, slug)
        if slug in solution_slugs:
            return f"](./leetgpu-{slug}-solution.html)"
        return match.group(0)

    return re.sub(
        r"\]\(https://leetgpu\.com/challenges/([a-z0-9-]+)\)",
        replace_link,
        markdown_text,
    )


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


def _build_nav(
    current_slug: Optional[str],
    solutions: List[Dict],
    current_topic: Optional[str] = None,
) -> str:
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
    topic_cls = "nav-link active" if current_topic == "cuda-interview-notes" else "nav-link"
    lines.append(f'<a class="{topic_cls}" href="./cuda-interview-notes.html">📝 CUDA 手撕题专题</a>')
    lines.append('<a class="nav-link" href="https://hzchenxiaobin.github.io/ai-infra-notes/index.html">📚 AI Infra 学习笔记</a>')
    lines.append('<a class="nav-link" href="https://hzchenxiaobin.github.io/leetcode/">🧩 LeetCode 题解</a>')
    return "\n".join(lines)


def _difficulty_descriptions() -> Dict[str, str]:
    return {
        "easy": "基础并行入门 —— 向量/矩阵运算、Element-wise 激活、简单归约",
        "medium": "工程进阶 —— 卷积/池化、扫描归约、归一化、Attention 变体、量化",
        "hard": "系统级挑战 —— 排序/FFT、图算法、完整 Transformer Block",
    }


def _build_landing_page(
    solutions: List[Dict],
    diff_groups: Dict[str, List[Dict]],
    difficulties: List[str],
    problems_json: str,
) -> str:
    """Generate the landing page HTML (sidebar-less, hero + cards layout)."""

    total = len(solutions)
    counts = {d: len(diff_groups.get(d, [])) for d in difficulties}

    diff_descs = _difficulty_descriptions()

    def _render_problem_cards(items: List[Dict]) -> str:
        cards = []
        for s in sorted(items, key=lambda x: x["number"]):
            cards.append(
                f'<a class="problem-card" href="./{s["slug"]}.html">'
                f'<span class="problem-card-badge">#{s["number"]}</span>'
                f'<span class="problem-card-name">{s["display_title"]}</span>'
                f'<span class="problem-card-arrow">→</span>'
                f'</a>'
            )
        return "\n".join(cards)

    difficulty_sections = []
    for d in difficulties:
        label = DIFFICULTY_LABELS.get(d, d)
        desc = diff_descs.get(d, "")
        count = counts.get(d, 0)
        diff_class = f"diff-{d}" if d in DIFFICULTY_ORDER else ""
        cards_html = _render_problem_cards(diff_groups.get(d, []))
        difficulty_sections.append(
            f'<div class="phase-group {diff_class}">\n'
            f'  <div class="phase-header">\n'
            f'    <span class="phase-no">{count} 题</span>\n'
            f'    <span class="phase-name">{label}</span>\n'
            f'    <span class="phase-desc">{desc}</span>\n'
            f'  </div>\n'
            f'  <div class="problem-grid">\n'
            f'{cards_html}\n'
            f'  </div>\n'
            f'</div>'
        )

    difficulty_html = "\n".join(difficulty_sections)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeetGPU 题解</title>
    <meta name="description" content="LeetGPU 题解：从基础 Kernel 到 Attention/GEMM，手写 CUDA 算子全覆盖，涵盖向量/矩阵运算、归约/排序、Transformer 算子、量化推理等高频考点。">
    <link rel="stylesheet" href="css/style.css?v=8">
</head>
<body class="landing">
    <header class="landing-nav">
        <a class="landing-nav-brand" href="./index.html">Leet<span>GPU</span></a>
        <nav class="landing-nav-links">
            <a href="#problems">题解</a>
            <a href="./cuda-interview-notes.html">CUDA 手撕题</a>
            <a class="landing-nav-github" href="https://github.com/hzchenxiaobin/leetgpu">GitHub ↗</a>
        </nav>
    </header>

    <section class="hero">
        <div class="hero-inner">
            <div class="hero-eyebrow">CUDA 工程实战 · 在线刷题</div>
            <h1 class="hero-title">LeetGPU <span class="hero-title-accent">题解</span></h1>
            <p class="hero-subtitle">从基础 Kernel 到 Attention/GEMM，手写 CUDA 算子全覆盖</p>
            <p class="hero-meta">涵盖向量/矩阵运算、归约/排序、Transformer 算子、量化推理等高频考点 · 每题含思路分析、Naive → Optimized 实现、性能对比</p>
            <div class="hero-actions">
                <button id="random-pick-btn" class="btn btn-primary" data-problems='{problems_json}'>🎲 随机选一道题</button>
                <a class="btn btn-secondary" href="#problems">📋 查看全部题解</a>
            </div>
        </div>
    </section>

    <section class="stats-strip">
        <div class="stat-item"><span class="stat-value">{total}</span><span class="stat-label">道题解</span></div>
        <div class="stat-item"><span class="stat-value">{counts.get("easy", 0)}</span><span class="stat-label">Easy</span></div>
        <div class="stat-item"><span class="stat-value">{counts.get("medium", 0)}</span><span class="stat-label">Medium</span></div>
        <div class="stat-item"><span class="stat-value">{counts.get("hard", 0)}</span><span class="stat-label">Hard</span></div>
    </section>

    <main class="landing-main">
        <section class="landing-section" id="problems">
            <h2 class="section-title">题解列表</h2>
            <p class="section-subtitle">按难度分组，点击卡片进入对应题解。每题包含思路拆解、Naive 实现、优化路径与 Profiling 对比。</p>
{difficulty_html}
        </section>

        <section class="landing-section">
            <h2 class="section-title">更多资源</h2>
            <div class="resource-grid">
                <a class="resource-card" href="./cuda-interview-notes.html">
                    <span class="resource-card-icon">📝</span>
                    <span class="resource-card-body">
                        <span class="resource-card-name">CUDA 手撕题专题</span>
                        <span class="resource-card-desc">面试高频 CUDA 手撕题与解析</span>
                    </span>
                </a>
                <a class="resource-card" href="https://hzchenxiaobin.github.io/ai-infra-notes/index.html">
                    <span class="resource-card-icon">📚</span>
                    <span class="resource-card-body">
                        <span class="resource-card-name">AI Infra 学习笔记</span>
                        <span class="resource-card-desc">10 周从 Kernel 到系统优化（独立站点）</span>
                    </span>
                </a>
                <a class="resource-card" href="https://hzchenxiaobin.github.io/leetcode/">
                    <span class="resource-card-icon">🧩</span>
                    <span class="resource-card-body">
                        <span class="resource-card-name">LeetCode 题解</span>
                        <span class="resource-card-desc">面试高频算法题解（独立站点）</span>
                    </span>
                </a>
                <a class="resource-card" href="https://github.com/hzchenxiaobin/leetgpu">
                    <span class="resource-card-icon">💻</span>
                    <span class="resource-card-body">
                        <span class="resource-card-name">GitHub 仓库</span>
                        <span class="resource-card-desc">本站的全部源码与 Markdown 原文</span>
                    </span>
                </a>
            </div>
        </section>
    </main>

    <footer class="landing-footer">
        <span>LeetGPU 题解 · 由 <a href="https://github.com/hzchenxiaobin/leetgpu">GitHub</a> 驱动 · Deployed on GitHub Pages</span>
    </footer>

    <button class="back-to-top" aria-label="Back to top">↑</button>
    <script>
    (function() {{
        var btn = document.getElementById('random-pick-btn');
        if (btn) {{
            btn.addEventListener('click', function() {{
                try {{
                    var problems = JSON.parse(btn.dataset.problems || '[]');
                    if (!problems.length) return;
                    var p = problems[Math.floor(Math.random() * problems.length)];
                    if (p.slug) {{
                        window.location.href = './leetgpu-' + p.slug + '-solution.html';
                    }}
                }} catch (e) {{}}
            }});
        }}
        var backTop = document.querySelector('.back-to-top');
        if (backTop) {{
            window.addEventListener('scroll', function() {{
                if (window.scrollY > 300) {{ backTop.classList.add('visible'); }}
                else {{ backTop.classList.remove('visible'); }}
            }});
            backTop.addEventListener('click', function() {{
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }});
        }}
    }})();
    </script>
</body>
</html>"""


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

    solution_slugs = {_challenge_slug(s["slug"]) for s in unique_solutions}
    for s in solutions:
        s["markdown"] = _rewrite_challenge_links(s["markdown"], solution_slugs)

    diff_groups: Dict[str, List[Dict]] = {}
    for s in solutions:
        d = s["difficulty"] or "未分组"
        diff_groups.setdefault(d, []).append(s)

    difficulties = sorted(diff_groups.keys(), key=_difficulty_sort_key)

    root_prefix = ""
    landing_html = _build_landing_page(solutions, diff_groups, difficulties, problems_json)
    (output_dir / "index.html").write_text(landing_html, encoding="utf-8")
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

    topic_md_path = LEETGPU_DIR / "cuda-interview-notes.md"
    if topic_md_path.exists():
        topic_text = topic_md_path.read_text(encoding="utf-8")
        topic_text = _rewrite_md_links_to_html(topic_text)
        topic_title = _parse_title(topic_text)
        topic_markdown = _strip_leading_h1(topic_text)
        topic_html = page_template(
            title=topic_title,
            nav_html=_build_nav(
                current_slug=None,
                solutions=solutions,
                current_topic="cuda-interview-notes",
            ),
            markdown=topic_markdown,
            root_prefix="",
            sidebar_title="LeetGPU 题解",
            sidebar_title_style="font-size: 1.5rem; margin-bottom: 0;",
            sidebar_href="./index.html",
            show_back_link=True,
            back_link_href="./index.html",
        )
        (output_dir / "cuda-interview-notes.html").write_text(topic_html, encoding="utf-8")
        print(f"Generated: {output_dir / 'cuda-interview-notes.html'}")
