/* ============================================================
   OpenDrug · homepage interactions
   ============================================================ */

(function () {
  "use strict";

  /* --------------------------------------------------------
     1. Sticky-nav scroll state
     -------------------------------------------------------- */
  const topbar = document.getElementById("topbar");
  const onScroll = () => {
    if (!topbar) return;
    if (window.scrollY > 16) topbar.classList.add("scrolled");
    else topbar.classList.remove("scrolled");
  };
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* --------------------------------------------------------
     2. Mobile nav toggle
     -------------------------------------------------------- */
  const navToggle = document.getElementById("navToggle");
  const nav = document.getElementById("nav");
  if (navToggle && nav) {
    navToggle.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      navToggle.classList.toggle("open", open);
      navToggle.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll("a").forEach((a) => {
      a.addEventListener("click", () => {
        nav.classList.remove("open");
        navToggle.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  /* --------------------------------------------------------
     3. Animated counters in hero
     -------------------------------------------------------- */
  const counters = document.querySelectorAll(".hero-stats .num");
  const animateCount = (el) => {
    const target = Number(el.dataset.target) || 0;
    const duration = 1400;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = String(Math.round(target * eased));
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        animateCount(entry.target);
        counterObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });
  counters.forEach((c) => counterObserver.observe(c));

  /* --------------------------------------------------------
     4. Hero network animation
        Small interactive heterogeneous graph:
        drug nodes (DDI/DTI/PPI) + protein nodes, soft edges.
     -------------------------------------------------------- */
  const canvas = document.getElementById("heroCanvas");
  if (canvas && canvas.getContext) {
    const ctx = canvas.getContext("2d");
    let dpr = window.devicePixelRatio || 1;
    let w = 0, h = 0;

    const COLORS = {
      bg: ["rgba(91,140,255,", "rgba(34,211,238,", "rgba(167,243,208,", "rgba(244,114,182,"],
      lines: "rgba(148, 175, 220, "
    };

    const rand = (a, b) => a + Math.random() * (b - a);
    const choice = (arr) => arr[Math.floor(Math.random() * arr.length)];

    let nodes = [];
    let edges = [];

    function buildGraph() {
      nodes = [];
      edges = [];
      const area = w * h;
      const density = Math.min(1, area / (1400 * 800));
      const count = Math.round(70 + 30 * density);

      // Drug-like nodes (circles)
      const drugs = [];
      for (let i = 0; i < count * 0.45; i++) {
        const n = {
          x: rand(0, w),
          y: rand(0, h),
          vx: rand(-0.18, 0.18),
          vy: rand(-0.18, 0.18),
          r: rand(2.2, 4.2),
          kind: "drug",
          color: choice(COLORS.bg),
          phase: rand(0, Math.PI * 2)
        };
        drugs.push(n);
        nodes.push(n);
      }
      // Protein-like nodes (triangles)
      for (let i = 0; i < count * 0.55; i++) {
        const n = {
          x: rand(0, w),
          y: rand(0, h),
          vx: rand(-0.14, 0.14),
          vy: rand(-0.14, 0.14),
          r: rand(2.8, 5.2),
          kind: "prot",
          color: choice(COLORS.bg),
          phase: rand(0, Math.PI * 2)
        };
        nodes.push(n);
      }

      // Build edges: link each node to nearest 2 nodes (one of each kind ideally)
      nodes.forEach((n) => {
        const sorted = nodes
          .filter((m) => m !== n)
          .map((m) => ({ m, d: Math.hypot(m.x - n.x, m.y - n.y) }))
          .sort((a, b) => a.d - b.d)
          .slice(0, 3);
        sorted.forEach(({ m }) => {
          const key = n.x < m.x ? `${n.x}|${n.y}-${m.x}|${m.y}` : `${m.x}|${m.y}-${n.x}|${n.y}`;
          if (!edges.find((e) => e.key === key)) {
            edges.push({ a: n, b: m, key, t: rand(0.08, 0.32) });
          }
        });
      });
    }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      buildGraph();
    }
    resize();
    window.addEventListener("resize", () => {
      dpr = window.devicePixelRatio || 1;
      resize();
    });

    let mouseX = -9999, mouseY = -9999;
    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left;
      mouseY = e.clientY - rect.top;
    });
    canvas.addEventListener("mouseleave", () => { mouseX = -9999; mouseY = -9999; });

    function step() {
      ctx.clearRect(0, 0, w, h);

      // Update positions
      nodes.forEach((n) => {
        n.x += n.vx;
        n.y += n.vy;
        // Mouse repulsion
        const dx = n.x - mouseX;
        const dy = n.y - mouseY;
        const dist = Math.hypot(dx, dy);
        if (dist < 110 && dist > 0) {
          const f = (110 - dist) / 110 * 0.05;
          n.x += (dx / dist) * f * 4;
          n.y += (dy / dist) * f * 4;
        }
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;
      });

      // Draw edges
      edges.forEach((e) => {
        const a = e.a, b = e.b;
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d = Math.hypot(dx, dy);
        if (d > 160) return;
        const op = e.t * (1 - d / 160) * 0.85;
        const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
        grad.addColorStop(0, a.color + op.toFixed(3) + ")");
        grad.addColorStop(1, b.color + op.toFixed(3) + ")");
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      });

      // Draw nodes
      nodes.forEach((n) => {
        n.phase += 0.015;
        const pulse = 0.8 + 0.2 * Math.sin(n.phase);
        ctx.fillStyle = n.color + (0.85 * pulse).toFixed(3) + ")";
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r * pulse, 0, Math.PI * 2);
        ctx.fill();
      });

      requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* --------------------------------------------------------
     5. Dataset filter + search
     -------------------------------------------------------- */
  const tabs = document.querySelectorAll(".tabs .tab");
  const rows = document.querySelectorAll("#datasetTable tbody tr");
  const search = document.getElementById("datasetSearch");

  let currentFilter = "all";

  const applyFilter = () => {
    const q = (search?.value || "").trim().toLowerCase();
    rows.forEach((row) => {
      const task = row.dataset.task || "";
      const matchesTask = currentFilter === "all" || task === currentFilter;
      const matchesQuery = !q || row.textContent.toLowerCase().includes(q);
      row.style.display = matchesTask && matchesQuery ? "" : "none";
    });
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      currentFilter = tab.dataset.filter || "all";
      applyFilter();
    });
  });
  if (search) search.addEventListener("input", applyFilter);

  /* --------------------------------------------------------
     6. Animate leaderboard bars when in view
     -------------------------------------------------------- */
  const bars = document.querySelectorAll(".lb-list .bar span");
  const barObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = el.style.width || "0%";
        el.style.width = "0%";
        requestAnimationFrame(() => {
          el.style.transition = "width 1.1s cubic-bezier(.2,.7,.2,1)";
          el.style.width = target;
        });
        barObserver.unobserve(el);
      }
    });
  }, { threshold: 0.3 });
  bars.forEach((b) => barObserver.observe(b));

  /* --------------------------------------------------------
     7. BibTeX copy
     -------------------------------------------------------- */
  const copyBtn = document.getElementById("copyBib");
  const toast = document.getElementById("copyToast");
  if (copyBtn && toast) {
    copyBtn.addEventListener("click", async () => {
      const targetId = copyBtn.dataset.target;
      const node = document.getElementById(targetId);
      if (!node) return;
      const text = node.innerText;
      try {
        await navigator.clipboard.writeText(text);
      } catch (e) {
        const range = document.createRange();
        range.selectNode(node);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand("copy");
        sel.removeAllRanges();
      }
      toast.classList.add("show");
      setTimeout(() => toast.classList.remove("show"), 1600);
    });
  }
})();