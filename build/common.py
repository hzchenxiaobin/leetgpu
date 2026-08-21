"""Shared utilities for the LeetGPU website build system."""

import shutil
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    text = text.replace("</script>", "\\x3c/script>")
    return text


def page_template(
    title: str,
    nav_html: str,
    markdown: str,
    *,
    root_prefix: str = "",
    page_title: Optional[str] = None,
    is_overview: bool = False,
    extra_scripts: str = "",
    sidebar_title: str = "LeetGPU 题解",
    sidebar_title_style: str = "",
    sidebar_href: Optional[str] = None,
    back_link_href: Optional[str] = None,
    show_back_link: bool = True,
    heading_renderer_js: str = "",
) -> str:
    """Generate a standard HTML page with sidebar navigation and markdown content."""
    escaped_markdown = escape_for_template_string(markdown)
    if page_title is None:
        page_title = title
    if sidebar_href is None:
        sidebar_href = f"{root_prefix}index.html"
    if back_link_href is None:
        back_link_href = f"{root_prefix}index.html"

    back_link = ""
    bottom_nav = ""
    if show_back_link and not is_overview:
        back_link = f'<a class="back-link" href="{back_link_href}">← 返回概览</a>'
        bottom_nav = f'<div class="day-nav-bottom"><a class="back-link" href="{back_link_href}">← 返回概览</a></div>'

    title_style_attr = f' style="{sidebar_title_style}"' if sidebar_title_style else ""

    if heading_renderer_js:
        renderer_block = f"        {heading_renderer_js}\n\n"
    else:
        renderer_block = ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <link rel="stylesheet" href="{root_prefix}css/style.css?v=7">
    <!-- Marked.js for Markdown rendering -->
    <script src="{root_prefix}js/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="{root_prefix}js/markdown-math.js"></script>
    <!-- Prism.js for syntax highlighting -->
    <link href="{root_prefix}css/prism-tomorrow.min.css" rel="stylesheet">
    <script src="{root_prefix}js/prism.min.js"></script>
    <script src="{root_prefix}js/prism-c.min.js"></script>
    <script src="{root_prefix}js/prism-cpp.min.js"></script>
    <script>Prism.languages.cuda=Prism.languages.extend("c",{{builtin:/\\b(?:__global__|__device__|__host__|__shared__|__constant__|__managed__|__restrict__|__syncthreads|__threadfence|__threadfence_block|blockIdx|threadIdx|blockDim|gridDim|warpSize)\\b/}});</script>
    <script src="{root_prefix}js/prism-bash.min.js"></script>
    <script src="{root_prefix}js/prism-python.min.js"></script>
    <!-- Restore collapsed sidebar before paint (desktop only) -->
    <script>(function(){{try{{if(localStorage.getItem('sidebar-collapsed')==='1'&&window.innerWidth>768){{document.documentElement.classList.add('sidebar-collapsed');}}}}catch(e){{}}}})();</script>
</head>
<body>
    <button class="menu-toggle" aria-label="Toggle menu">☰</button>

    <div class="site-container">
        <aside class="sidebar">
            <div class="sidebar-header">
                <a href="{sidebar_href}" style="text-decoration: none;">
                    <h1 class="sidebar-title"{title_style_attr}>{sidebar_title}</h1>
                </a>
            </div>
            <nav class="sidebar-nav">
{nav_html}
            </nav>
        </aside>

        <main class="main-content">
            <div class="page-header">
                <h1 class="page-title">{title}</h1>
                {back_link}
            </div>
            <article class="content" id="content"></article>
            {bottom_nav}
        </main>
    </div>

    <button class="back-to-top" aria-label="Back to top">↑</button>

    <script>
        const markdown = `{escaped_markdown}`;

        const renderer = new marked.Renderer();
{renderer_block}        marked.setOptions({{
            renderer: renderer,
            headerIds: false,
            gfm: true,
            breaks: false,
            sanitize: false
        }});

        try {{
            if (typeof marked === 'undefined') {{
                throw new Error('marked.js failed to load. Please check js/marked.min.js exists.');
            }}
            document.getElementById('content').innerHTML = marked.parse(markdown);

            if (window.Prism) {{
                Prism.highlightAll();
            }}
        }} catch (err) {{
            document.getElementById('content').innerHTML = '<div style="padding: 20px; color: #ff7b72; background: #2d1515; border-radius: 8px;">' +
                '<h2>⚠️ 页面渲染失败</h2>' +
                '<p>' + err.message + '</p>' +
                '<p>请打开浏览器控制台（Cmd + Option + J）查看详细错误。</p>' +
                '</div>';
            console.error('Markdown render error:', err);
        }}
    </script>
    {extra_scripts}
    <script src="{root_prefix}js/main.js?v=6"></script>
</body>
</html>
"""


def copy_static_assets(public_dir: Path) -> None:
    """Copy shared css/js from static/ to public/css/ and public/js/."""
    css_src = STATIC_DIR / "css"
    js_src = STATIC_DIR / "js"
    if css_src.exists():
        dst = public_dir / "css"
        dst.mkdir(parents=True, exist_ok=True)
        for item in css_src.iterdir():
            if item.is_file():
                shutil.copy2(item, dst / item.name)
    if js_src.exists():
        dst = public_dir / "js"
        dst.mkdir(parents=True, exist_ok=True)
        for item in js_src.iterdir():
            if item.is_file():
                shutil.copy2(item, dst / item.name)
