document.addEventListener('DOMContentLoaded', function() {
    // Scroll the active day pill into view inside the top nav (mobile)
    const activePill = document.querySelector('.landing-nav-pills .day-pill-active');
    if (activePill) {
        activePill.scrollIntoView({ inline: 'center', block: 'nearest' });
    }

    // Mobile menu toggle
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');

    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function() {
            if (window.innerWidth <= 768) {
                // Mobile: sidebar is an overlay
                sidebar.classList.toggle('open');
            } else {
                // Desktop: collapse/expand sidebar to give content more room
                const collapsed = document.documentElement.classList.toggle('sidebar-collapsed');
                try {
                    localStorage.setItem('sidebar-collapsed', collapsed ? '1' : '0');
                } catch (e) {}
            }
        });

        // Close sidebar when clicking on a link (mobile)
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    sidebar.classList.remove('open');
                }
            });
        });
    }

    // Scroll sidebar independently on wheel when it has overflow
    if (sidebar) {
        sidebar.addEventListener('wheel', function(e) {
            if (e.deltaY === 0) return;
            const canScrollUp = sidebar.scrollTop > 0;
            const canScrollDown = sidebar.scrollTop + sidebar.clientHeight < sidebar.scrollHeight;
            if ((e.deltaY < 0 && canScrollUp) || (e.deltaY > 0 && canScrollDown)) {
                e.preventDefault();
                sidebar.scrollTop += e.deltaY;
            }
        }, { passive: false });
    }

    // Draggable sidebar width resizer
    initSidebarResizer();

    // Accordion navigation in sidebar
    function toggleAccordionItem(item) {
        if (!item) return;
        const willExpand = !item.classList.contains('is-expanded');
        item.classList.toggle('is-expanded');
        const isExpanded = item.classList.contains('is-expanded');
        const button = item.querySelector('.nav-accordion-toggle');
        if (button) {
            button.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
        }
        // Animate max-height with real measurements so long lists are never
        // clipped by a fixed cap. After the expand transition finishes, the
        // inline style is cleared and the CSS `max-height: none` rule applies.
        const content = item.querySelector(':scope > .nav-accordion-content');
        if (content) {
            if (isExpanded) {
                content.style.maxHeight = content.scrollHeight + 'px';
                content.addEventListener('transitionend', function handler(e) {
                    if (e.propertyName !== 'max-height') return;
                    content.removeEventListener('transitionend', handler);
                    if (item.classList.contains('is-expanded')) {
                        content.style.maxHeight = '';
                    }
                });
            } else {
                content.style.maxHeight = content.scrollHeight + 'px';
                void content.offsetHeight; // force reflow so the collapse animates
                content.style.maxHeight = '0px';
            }
        }
        // When manually expanding a level, collapse its descendants so that
        // each level requires its own click to expand.
        if (willExpand) {
            item.querySelectorAll('.nav-accordion-item.is-expanded').forEach(child => {
                child.classList.remove('is-expanded');
                const childButton = child.querySelector('.nav-accordion-toggle');
                if (childButton) {
                    childButton.setAttribute('aria-expanded', 'false');
                }
                const childContent = child.querySelector(':scope > .nav-accordion-content');
                if (childContent) {
                    childContent.style.maxHeight = '';
                }
            });
        }
    }

    document.querySelectorAll('.nav-accordion-toggle').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleAccordionItem(button.closest('.nav-accordion-item'));
        });
    });

    // Clicking a week header toggles the accordion; week links don't navigate.
    document.querySelectorAll('.nav-accordion-header').forEach(header => {
        header.addEventListener('click', function(e) {
            const link = e.target.closest('.week-link');
            if (link) {
                e.preventDefault();
                e.stopPropagation();
            }
            // Don't toggle if the click was on a day-link inside the header (future-proofing)
            if (e.target.closest('.day-link')) {
                return;
            }
            toggleAccordionItem(header.closest('.nav-accordion-item'));
            // Prevent nested accordion clicks from toggling parent accordions
            e.stopPropagation();
        });
    });

    // Add copy buttons to code blocks
    document.querySelectorAll('.content pre').forEach(pre => {
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);

        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.addEventListener('click', async function() {
            const code = pre.querySelector('code');
            const text = code ? code.textContent : pre.textContent;
            try {
                await navigator.clipboard.writeText(text);
                button.textContent = 'Copied!';
                button.classList.add('copied');
                setTimeout(() => {
                    button.textContent = 'Copy';
                    button.classList.remove('copied');
                }, 2000);
            } catch (err) {
                button.textContent = 'Failed';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            }
        });
        wrapper.appendChild(button);
    });

    // Back to top button
    const backToTop = document.querySelector('.back-to-top');
    if (backToTop) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 300) {
                backToTop.classList.add('visible');
            } else {
                backToTop.classList.remove('visible');
            }
        });

        backToTop.addEventListener('click', function() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // Random LeetGPU problem picker on overview page
    const randomPickBtn = document.getElementById('random-pick-btn');
    if (randomPickBtn) {
        randomPickBtn.addEventListener('click', function() {
            try {
                const problems = JSON.parse(randomPickBtn.dataset.problems || '[]');
                if (!problems.length) return;
                const p = problems[Math.floor(Math.random() * problems.length)];
                if (p.url) {
                    window.open(p.url, '_blank', 'noopener,noreferrer');
                } else if (p.slug) {
                    window.open('https://leetgpu.com/challenges/' + encodeURIComponent(p.slug), '_blank',
                                'noopener,noreferrer');
                }
            } catch (e) {
                // Ignore malformed data attribute
            }
        });
    }

    // Image lightbox zoom
    initImageLightbox();

    // Enhance interview Q&A section into styled cards
    enhanceInterviewQA();

    // Open external links in new tab; on non-landing pages, open all non-sidebar
    // links in new tab (existing behavior). On the landing page, only external
    // links get target="_blank" so internal navigation stays in-tab.
    document.querySelectorAll('a').forEach(link => {
        if (link.closest('.sidebar')) {
            return;
        }
        if (document.body.classList.contains('landing')) {
            const href = link.getAttribute('href') || '';
            if (href.startsWith('http://') || href.startsWith('https://')) {
                try {
                    if (new URL(href).origin !== window.location.origin) {
                        link.setAttribute('target', '_blank');
                        link.setAttribute('rel', 'noopener noreferrer');
                    }
                } catch (e) {}
            }
        } else {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        }
    });
});

function enhanceInterviewQA() {
    const content = document.querySelector('.content');
    if (!content) return;

    content.querySelectorAll('h4').forEach(heading => {
        const details = heading.nextElementSibling;
        if (!details || details.tagName !== 'DETAILS') return;

        // Avoid re-processing
        if (heading.closest('.qa-card')) return;

        const card = document.createElement('div');
        card.className = 'qa-card';

        heading.classList.add('qa-question');
        details.classList.add('qa-answer');

        const summary = details.querySelector('summary');
        if (summary) {
            summary.classList.add('qa-answer-toggle');
        }

        heading.parentNode.insertBefore(card, heading);
        card.appendChild(heading);
        card.appendChild(details);
    });
}

function initImageLightbox() {
    const content = document.querySelector('.content');
    if (!content) return;

    const images = content.querySelectorAll('img');
    if (images.length === 0) return;

    const lightbox = document.createElement('div');
    lightbox.className = 'image-lightbox';
    lightbox.setAttribute('role', 'dialog');
    lightbox.setAttribute('aria-modal', 'true');
    lightbox.setAttribute('aria-label', 'Image preview');
    lightbox.innerHTML = `
        <button class="lightbox-close" aria-label="Close image preview">&times;</button>
        <img src="" alt="">
        <div class="lightbox-caption"></div>
        <div class="lightbox-zoom-hint">滚轮缩放 / 点击关闭</div>
    `;
    document.body.appendChild(lightbox);

    const lightboxImg = lightbox.querySelector('img');
    const lightboxCaption = lightbox.querySelector('.lightbox-caption');
    const closeButton = lightbox.querySelector('.lightbox-close');
    const zoomHint = lightbox.querySelector('.lightbox-zoom-hint');

    let currentScale = 1;
    let currentTranslateX = 0;
    let currentTranslateY = 0;
    const MIN_SCALE = 0.5;
    const MAX_SCALE = 5;
    const ZOOM_STEP = 0.15;

    function updateTransform() {
        lightboxImg.style.transform = `translate(calc(-50% + ${currentTranslateX}px), calc(-50% + ${currentTranslateY}px)) scale(${currentScale})`;
    }

    function updateScale(scale) {
        currentScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
        updateTransform();
    }

    function resetScale() {
        currentScale = 1;
        currentTranslateX = 0;
        currentTranslateY = 0;
        lightboxImg.style.transform = '';
    }

    function openLightbox(img) {
        resetScale();
        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt || '';
        if (img.alt) {
            lightboxCaption.textContent = img.alt;
            lightboxCaption.style.display = 'block';
        } else {
            lightboxCaption.style.display = 'none';
        }
        lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = '';
        resetScale();
    }

    images.forEach(img => {
        img.style.cursor = 'pointer';
        img.addEventListener('click', function(e) {
            e.preventDefault();
            openLightbox(img);
        });
    });

    lightbox.addEventListener('click', function(e) {
        if (hasDragged) return;
        if (e.target === lightbox || e.target === lightboxImg) {
            closeLightbox();
        }
    });

    closeButton.addEventListener('click', closeLightbox);

    // Drag to pan the zoomed image
    let isDragging = false;
    let hasDragged = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let dragStartTranslateX = 0;
    let dragStartTranslateY = 0;

    function clampPan() {
        const rect = lightboxImg.getBoundingClientRect();
        const viewportWidth = lightbox.clientWidth;
        const viewportHeight = lightbox.clientHeight;
        const halfWidth = rect.width / 2;
        const halfHeight = rect.height / 2;

        // Allow the image center to move within the viewport plus half its size,
        // so users can drag any part of the image into view.
        const maxX = Math.max(0, halfWidth + viewportWidth / 2);
        const maxY = Math.max(0, halfHeight + viewportHeight / 2);

        currentTranslateX = Math.max(-maxX, Math.min(maxX, currentTranslateX));
        currentTranslateY = Math.max(-maxY, Math.min(maxY, currentTranslateY));
    }

    lightboxImg.addEventListener('mousedown', function(e) {
        if (!lightbox.classList.contains('active')) return;
        isDragging = true;
        hasDragged = false;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        dragStartTranslateX = currentTranslateX;
        dragStartTranslateY = currentTranslateY;
        lightboxImg.classList.add('dragging');
        e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
        if (!isDragging) return;
        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
            hasDragged = true;
        }
        currentTranslateX = dragStartTranslateX + dx;
        currentTranslateY = dragStartTranslateY + dy;
        clampPan();
        updateTransform();
    });

    document.addEventListener('mouseup', function() {
        if (!isDragging) return;
        isDragging = false;
        lightboxImg.classList.remove('dragging');
        setTimeout(() => { hasDragged = false; }, 50);
    });

    // Touch support for mobile
    lightboxImg.addEventListener('touchstart', function(e) {
        if (!lightbox.classList.contains('active')) return;
        if (e.touches.length !== 1) return;
        isDragging = true;
        hasDragged = false;
        dragStartX = e.touches[0].clientX;
        dragStartY = e.touches[0].clientY;
        dragStartTranslateX = currentTranslateX;
        dragStartTranslateY = currentTranslateY;
        lightboxImg.classList.add('dragging');
    }, { passive: false });

    document.addEventListener('touchmove', function(e) {
        if (!isDragging) return;
        if (e.touches.length !== 1) return;
        const dx = e.touches[0].clientX - dragStartX;
        const dy = e.touches[0].clientY - dragStartY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
            hasDragged = true;
        }
        currentTranslateX = dragStartTranslateX + dx;
        currentTranslateY = dragStartTranslateY + dy;
        clampPan();
        updateTransform();
        e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchend', function() {
        if (!isDragging) return;
        isDragging = false;
        lightboxImg.classList.remove('dragging');
        setTimeout(() => { hasDragged = false; }, 50);
    });

    // Mouse wheel zoom
    lightbox.addEventListener('wheel', function(e) {
        if (!lightbox.classList.contains('active')) return;
        e.preventDefault();
        const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
        updateScale(currentScale + delta);
    }, { passive: false });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && lightbox.classList.contains('active')) {
            closeLightbox();
        }
    });
}


function initSidebarResizer() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    if (!sidebar || !mainContent) return;

    // Don't show resizer on mobile where sidebar is an overlay
    if (window.innerWidth <= 768) return;

    const resizer = document.createElement('div');
    resizer.className = 'sidebar-resizer';
    resizer.setAttribute('aria-label', '调整侧边栏宽度');
    document.body.appendChild(resizer);

    const MIN_WIDTH = 220;
    const MAX_WIDTH = 520;
    const DEFAULT_WIDTH = 300;

    function setSidebarWidth(width) {
        sidebar.style.width = width + 'px';
        mainContent.style.marginLeft = width + 'px';
        resizer.style.left = width + 'px';
    }

    // Restore saved width on load
    const savedWidth = localStorage.getItem('sidebar-width');
    if (savedWidth) {
        const width = parseInt(savedWidth, 10);
        if (!isNaN(width) && width >= MIN_WIDTH && width <= MAX_WIDTH) {
            setSidebarWidth(width);
        }
    }

    let isResizing = false;

    resizer.addEventListener('mousedown', function(e) {
        isResizing = true;
        resizer.classList.add('resizing');
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'col-resize';
        e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
        if (!isResizing) return;
        const newWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, e.clientX));
        setSidebarWidth(newWidth);
    });

    document.addEventListener('mouseup', function() {
        if (!isResizing) return;
        isResizing = false;
        resizer.classList.remove('resizing');
        document.body.style.userSelect = '';
        document.body.style.cursor = '';
        localStorage.setItem('sidebar-width', sidebar.offsetWidth);
    });

    // Clean up on orientation/resize changes to mobile layout
    window.addEventListener('resize', function() {
        if (window.innerWidth <= 768) {
            resizer.style.display = 'none';
            sidebar.style.width = '';
            mainContent.style.marginLeft = '';
        } else {
            resizer.style.display = '';
            const currentWidth = sidebar.offsetWidth;
            if (currentWidth < MIN_WIDTH || currentWidth > MAX_WIDTH) {
                setSidebarWidth(DEFAULT_WIDTH);
            } else {
                resizer.style.left = currentWidth + 'px';
            }
        }
    });
}
