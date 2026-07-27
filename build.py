#!/usr/bin/env python3
"""
Build the LeetGPU solutions website for GitHub Pages.
Generates public/ (deployment root):
  - Shared css/js (copied from static/)
  - LeetGPU overview index + flat solution pages (built by build.leetgpu)
"""

import shutil
from pathlib import Path

from build.common import copy_static_assets
from build.leetgpu import build as build_leetgpu


def main() -> None:
    repo_root = Path(__file__).parent
    public_dir = repo_root / "public"

    if public_dir.exists():
        shutil.rmtree(public_dir)
    public_dir.mkdir()

    print("Copying static assets (css/js)...")
    copy_static_assets(public_dir)

    print("Building LeetGPU website...")
    build_leetgpu(public_dir)

    print("Website built successfully in public/")


if __name__ == "__main__":
    main()
