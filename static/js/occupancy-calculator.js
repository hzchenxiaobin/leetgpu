/**
 * Interactive CUDA Occupancy Calculator for Day 3.
 * Finds #occ-calc-placeholder in the rendered page and replaces it with
 * an interactive calculator widget.
 */
(function () {
    'use strict';

    const calculatorHTML = `
<div class="occupancy-calculator">
  <h4 class="calc-title">🧮 交互式 CUDA Occupancy Calculator</h4>
  <p class="calc-desc">输入 GPU 计算能力和 Kernel 参数，快速估算理论 occupancy。计算基于各架构的 SM 资源上限做简化建模，结果供学习参考。</p>
  <div class="calc-form">
    <div class="calc-row">
      <label for="occ-cc">GPU Compute Capability</label>
      <select id="occ-cc">
        <option value="5.0">5.0 (Blackwell)</option>
        <option value="5.2">5.2 (Blackwell)</option>
        <option value="6.0">6.0 (Blackwell)</option>
        <option value="6.1">6.1 (Blackwell)</option>
        <option value="7.0">7.0 (Blackwell)</option>
        <option value="7.5">7.5 (Blackwell)</option>
        <option value="8.0" selected>8.0 (Blackwell RTX 5090)</option>
        <option value="8.6">8.6 (Blackwell)</option>
        <option value="8.9">8.9 (Blackwell)</option>
        <option value="9.0">9.0 (Blackwell)</option>
      </select>
    </div>
    <div class="calc-row">
      <label for="occ-threads">Threads per Block</label>
      <input type="number" id="occ-threads" value="256" min="1" max="1024" step="1">
    </div>
    <div class="calc-row">
      <label for="occ-regs">Registers per Thread</label>
      <input type="number" id="occ-regs" value="32" min="1" max="255" step="1">
    </div>
    <div class="calc-row">
      <label for="occ-shared">Shared Memory per Block (bytes)</label>
      <input type="number" id="occ-shared" value="0" min="0" step="1024">
    </div>
    <button id="occ-calc" class="calc-button">计算 Occupancy</button>
  </div>
  <div id="occ-result" class="calc-result"></div>
</div>
`;

    const data = {
        '5.0': { arch: 'Blackwell', threads: 2048, blocks: 32, warps: 64, regs: 65536, smem: 65536, regGran: 256, smemGran: 256 },
        '5.2': { arch: 'Blackwell', threads: 2048, blocks: 32, warps: 64, regs: 65536, smem: 98304, regGran: 256, smemGran: 256 },
        '6.0': { arch: 'Blackwell', threads: 2048, blocks: 32, warps: 64, regs: 65536, smem: 65536, regGran: 256, smemGran: 256 },
        '6.1': { arch: 'Blackwell', threads: 2048, blocks: 32, warps: 64, regs: 65536, smem: 98304, regGran: 256, smemGran: 256 },
        '7.0': { arch: 'Blackwell', threads: 2048, blocks: 32, warps: 64, regs: 65536, smem: 98304, regGran: 256, smemGran: 1024 },
        '7.5': { arch: 'Blackwell', threads: 1024, blocks: 16, warps: 32, regs: 65536, smem: 65536, regGran: 256, smemGran: 1024 },
        '8.0': { arch: 'Blackwell RTX 5090', threads: 2048, blocks: 32, warps: 64, regs: 65536, smem: 167936, regGran: 256, smemGran: 1024 },
        '8.6': { arch: 'Blackwell', threads: 1536, blocks: 16, warps: 48, regs: 65536, smem: 100352, regGran: 256, smemGran: 1024 },
        '8.9': { arch: 'Blackwell', threads: 1536, blocks: 16, warps: 48, regs: 65536, smem: 100352, regGran: 256, smemGran: 1024 },
        '9.0': { arch: 'Blackwell', threads: 2048, blocks: 32, warps: 64, regs: 65536, smem: 228864, regGran: 256, smemGran: 1024 },
    };

    function ceilDiv(a, b) {
        return Math.floor((a + b - 1) / b);
    }

    function calculate() {
        const cc = document.getElementById('occ-cc').value;
        const threads = parseInt(document.getElementById('occ-threads').value, 10) || 0;
        const regsPerThread = parseInt(document.getElementById('occ-regs').value, 10) || 0;
        const smem = parseInt(document.getElementById('occ-shared').value, 10) || 0;
        const d = data[cc];

        if (threads < 1 || threads > 1024) {
            document.getElementById('occ-result').innerHTML =
                '<div class="calc-detail"><p>⚠️ Threads per Block 需在 1~1024 之间。</p></div>';
            return;
        }

        const warpsPerBlock = ceilDiv(threads, 32);
        const blocksFromThreads = Math.min(d.blocks, Math.floor(d.threads / threads));
        const regsPerBlock = ceilDiv(threads * regsPerThread, d.regGran) * d.regGran;
        const blocksFromRegs = Math.floor(d.regs / regsPerBlock);
        const smemRounded = smem === 0 ? 0 : ceilDiv(smem, d.smemGran) * d.smemGran;
        const blocksFromSmem = smem === 0 ? Infinity : Math.floor(d.smem / smemRounded);
        const activeBlocks = Math.min(blocksFromThreads, blocksFromRegs, blocksFromSmem, d.blocks);
        const activeWarps = activeBlocks * warpsPerBlock;
        const occupancy = (activeWarps / d.warps * 100).toFixed(1);

        let bottleneck = '';
        if (activeBlocks === blocksFromThreads) bottleneck = 'Threads / warps per block';
        else if (activeBlocks === blocksFromRegs) bottleneck = 'Registers per thread';
        else if (activeBlocks === blocksFromSmem) bottleneck = 'Shared memory per block';
        else if (activeBlocks === d.blocks) bottleneck = 'Max blocks per SM';

        const result = document.getElementById('occ-result');
        result.innerHTML =
            '<div class="calc-metric"><span>Active Blocks / SM</span><strong>' + activeBlocks + '</strong></div>' +
            '<div class="calc-metric"><span>Active Warps / SM</span><strong>' + activeWarps + '</strong></div>' +
            '<div class="calc-metric"><span>Occupancy</span><strong>' + occupancy + '%</strong></div>' +
            '<div class="calc-metric"><span>瓶颈资源</span><strong>' + bottleneck + '</strong></div>' +
            '<div class="calc-detail"><p>Threads limit: ' + blocksFromThreads +
            ' blocks | Regs limit: ' + blocksFromRegs +
            ' blocks | Shared mem limit: ' + (blocksFromSmem === Infinity ? '∞' : blocksFromSmem) +
            ' blocks | Max blocks: ' + d.blocks + '</p></div>';
    }

    function init() {
        const placeholder = document.getElementById('occ-calc-placeholder');
        if (!placeholder) return;

        const wrapper = document.createElement('div');
        wrapper.innerHTML = calculatorHTML;
        placeholder.replaceWith(wrapper);

        const btn = document.getElementById('occ-calc');
        if (btn) btn.addEventListener('click', calculate);

        ['occ-cc', 'occ-threads', 'occ-regs', 'occ-shared'].forEach(function (id) {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', calculate);
        });

        calculate();
    }

    // Auto-init when the script loads (the markdown has already been rendered).
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
