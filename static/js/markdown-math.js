/**
 * markdown-math.js
 * 为 marked.js (v12.x) 增加 $...$ / $$...$$ 公式支持，并用 KaTeX 渲染。
 *
 * 使用方式：在 marked.min.js 之后、调用 marked.parse() 之前加载本文件。
 * 本文件会注册 marked 扩展，将公式保留为 <span class="math-inline"> /
 * <div class="math-block"> 占位元素；页面加载完成后自动调用 KaTeX 渲染。
 */
(function () {
    'use strict';

    function escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    /**
     * 清理 LaTeX 源码：去掉首尾的空白和换行，统一换行符。
     */
    function normalizeLatex(latex) {
        return latex.replace(/\r\n/g, '\n').replace(/^\n+|\n+$/g, '').trim();
    }

    /**
     * 注册 marked 扩展。使用扩展可以让 marked 在解析代码块、列表等结构时
     * 自动跳过内部的 $ 符号，避免代码中的 $ 被误识别为公式。
     */
    function registerMarkedExtensions() {
        if (typeof marked === 'undefined' || !marked.use) {
            console.warn('[markdown-math] marked.js not found, math extensions not registered.');
            return;
        }

        const mathBlock = {
            name: 'mathBlock',
            level: 'block',
            start(src) {
                return src.match(/\$\$/) ? src.indexOf('$$') : -1;
            },
            tokenizer(src) {
                // 匹配 $$...$$，允许跨行
                const rule = /^\$\$\n?([\s\S]+?)\n?\$\$/;
                const match = rule.exec(src);
                if (match) {
                    const latex = normalizeLatex(match[1]);
                    return {
                        type: 'html',
                        raw: match[0],
                        text: '<div class="math-block" data-latex="' + escapeHtml(latex) + '"></div>'
                    };
                }
                return undefined;
            }
        };

        const mathInline = {
            name: 'mathInline',
            level: 'inline',
            start(src) {
                const idx = src.indexOf('$');
                if (idx === -1) return -1;
                // 避免 $$ 开头时被 inline 扩展先吃掉
                if (src.charAt(idx + 1) === '$') return -1;
                return idx;
            },
            tokenizer(src) {
                // 匹配 $...$，内容不能包含 $ 和换行
                const rule = /^\$([^\$\n]+?)\$/;
                const match = rule.exec(src);
                if (match) {
                    const latex = match[1].trim();
                    return {
                        type: 'html',
                        raw: match[0],
                        text: '<span class="math-inline" data-latex="' + escapeHtml(latex) + '"></span>'
                    };
                }
                return undefined;
            }
        };

        marked.use({ extensions: [mathBlock, mathInline] });
    }

    /**
     * 渲染页面中所有公式占位元素。
     */
    function renderMath() {
        if (typeof katex === 'undefined') {
            console.warn('[markdown-math] KaTeX not loaded, skipping math render.');
            return;
        }

        document.querySelectorAll('.math-inline').forEach(function (el) {
            try {
                katex.render(el.dataset.latex, el, {
                    throwOnError: false,
                    displayMode: false
                });
            } catch (e) {
                console.error('[markdown-math] inline math render error:', e);
            }
        });

        document.querySelectorAll('.math-block').forEach(function (el) {
            try {
                katex.render(el.dataset.latex, el, {
                    throwOnError: false,
                    displayMode: true
                });
            } catch (e) {
                console.error('[markdown-math] block math render error:', e);
            }
        });
    }

    // 立即注册扩展
    registerMarkedExtensions();

    // DOM 准备好后渲染公式；如果 KaTeX 是同步加载的，这里可以直接渲染。
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', renderMath);
    } else {
        // DOM 已就绪，立即执行
        renderMath();
    }
})();
