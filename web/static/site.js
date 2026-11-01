(function () {
  "use strict";

  var doc = document.documentElement;
  var root = doc.dataset.root || "./";
  var main = document.querySelector("[data-lesson-id]");

  // Storage can throw or be empty (private windows, blocked site data); the site
  // must keep working without it, so every access goes through these two.
  function load(key, fallback) {
    try {
      var raw = localStorage.getItem("fsds:" + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (e) {
      return fallback;
    }
  }
  function save(key, value) {
    try {
      localStorage.setItem("fsds:" + key, JSON.stringify(value));
    } catch (e) {}
  }

  /* Theme ---------------------------------------------------------------- */
  document.querySelectorAll(".theme-toggle").forEach(function (toggle) {
    toggle.addEventListener("click", function () {
      var current = doc.getAttribute("data-theme");
      if (!current) {
        current = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      }
      var next = current === "dark" ? "light" : "dark";
      doc.setAttribute("data-theme", next);
      try {
        localStorage.setItem("fsds:theme", next);
      } catch (e) {}
    });
  });

  /* Mobile navigation ---------------------------------------------------- */
  var navToggle = document.querySelector(".nav-toggle");
  var scrim = document.querySelector(".nav-scrim");
  function setNav(open) {
    document.body.classList.toggle("nav-open", open);
    navToggle.setAttribute("aria-expanded", String(open));
    navToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    scrim.hidden = !open;
  }
  if (navToggle) {
    navToggle.addEventListener("click", function () {
      setNav(!document.body.classList.contains("nav-open"));
    });
    scrim.addEventListener("click", function () {
      setNav(false);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") setNav(false);
    });
  }
  var current = document.querySelector(".sidebar .is-current");
  if (current) current.scrollIntoView({ block: "center" });

  /* Progress ------------------------------------------------------------- */
  function renderProgress() {
    var done = load("progress", {});
    document.querySelectorAll("[data-lesson-link]").forEach(function (link) {
      link.classList.toggle("is-complete", Boolean(done[link.dataset.lessonLink]));
    });
    var ids = Object.keys(done);
    document.querySelectorAll(".progress").forEach(function (bar) {
      var count = ids.filter(function (id) {
        return id.indexOf(bar.dataset.module + "/") === 0;
      }).length;
      var share = Math.min(100, (100 * count) / Number(bar.dataset.total));
      bar.firstElementChild.style.width = share + "%";
      bar.setAttribute("role", "img");
      bar.setAttribute("aria-label", count + " of " + bar.dataset.total + " lessons complete");
    });
    var total = document.querySelector(".total-progress");
    if (total) total.textContent = ids.length + " of " + total.dataset.total;
  }

  if (main) {
    save("last", {
      title: main.dataset.lessonTitle,
      path: main.dataset.lessonPath,
    });
    var button = document.querySelector(".complete-button");
    var id = main.dataset.lessonId;
    var paint = function () {
      var isDone = Boolean(load("progress", {})[id]);
      button.classList.toggle("is-complete", isDone);
      button.setAttribute("aria-checked", String(isDone));
      button.querySelector(".complete-label").textContent = isDone
        ? "Lesson completed"
        : "Mark lesson as complete";
    };
    button.addEventListener("click", function () {
      var done = load("progress", {});
      if (done[id]) delete done[id];
      else done[id] = true;
      save("progress", done);
      paint();
      renderProgress();
    });
    paint();
  }
  renderProgress();

  var continueLink = document.querySelector(".continue-link");
  var last = load("last", null);
  if (continueLink && last && last.path) {
    continueLink.href = root + last.path;
    continueLink.textContent = "Continue: " + last.title;
  }

  /* Self-assessment checklists ------------------------------------------- */
  var checksKey = "checks:" + doc.dataset.page;
  var boxes = document.querySelectorAll(".task-list-item-checkbox");
  if (boxes.length) {
    var checked = load(checksKey, {});
    boxes.forEach(function (box, index) {
      box.disabled = false;
      box.checked = Boolean(checked[index]);
      box.addEventListener("change", function () {
        var state = load(checksKey, {});
        state[index] = box.checked;
        save(checksKey, state);
      });
    });
  }

  /* Copy buttons --------------------------------------------------------- */
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    // file:// pages have no async clipboard.
    var area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    var ok = document.execCommand("copy");
    area.remove();
    return ok ? Promise.resolve() : Promise.reject(new Error("copy failed"));
  }
  document.querySelectorAll(".copy").forEach(function (button) {
    button.addEventListener("click", function () {
      var code = button.parentElement.querySelector("code").textContent;
      copyText(code).then(
        function () {
          button.textContent = "Copied";
        },
        function () {
          button.textContent = "Press Ctrl+C";
        }
      );
      setTimeout(function () {
        button.textContent = "Copy";
      }, 1600);
    });
  });

  /* Math ----------------------------------------------------------------- */
  if (window.katex) {
    document.querySelectorAll(".tex").forEach(function (node) {
      window.katex.render(node.textContent, node, {
        displayMode: node.dataset.display === "true",
        throwOnError: false,
      });
    });
  }

  /* Table of contents ---------------------------------------------------- */
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    var byId = {};
    tocLinks.forEach(function (link) {
      byId[decodeURIComponent(link.hash.slice(1))] = link;
    });
    var visible = new Set();
    var headings = Object.keys(byId)
      .map(function (key) {
        return document.getElementById(key);
      })
      .filter(Boolean);
    var markActive = function () {
      var active = headings.filter(function (heading) {
        return visible.has(heading);
      })[0];
      if (!active) return;
      tocLinks.forEach(function (link) {
        var isActive = link === byId[active.id];
        link.classList.toggle("is-active", isActive);
        if (isActive) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
      var toc = document.querySelector(".toc");
      var link = byId[active.id];
      if (link.offsetTop < toc.scrollTop || link.offsetTop > toc.scrollTop + toc.clientHeight - 60) {
        toc.scrollTop = link.offsetTop - toc.clientHeight / 2;
      }
    };
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) visible.add(entry.target);
          else visible.delete(entry.target);
        });
        markActive();
      },
      { rootMargin: "-60px 0px -70% 0px" }
    );
    headings.forEach(function (heading) {
      observer.observe(heading);
    });
  }

  /* Search --------------------------------------------------------------- */
  var dialog = document.querySelector(".search-dialog");
  var input = dialog.querySelector(".search-input");
  var results = dialog.querySelector(".search-results");
  var indexRequested = false;
  var selected = -1;

  function loadIndex() {
    if (indexRequested) return;
    indexRequested = true;
    var script = document.createElement("script");
    script.src = root + "static/search-index.js";
    script.onload = runSearch;
    document.head.appendChild(script);
  }

  function escapeHtml(text) {
    return text.replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function snippet(text, terms) {
    var lower = text.toLowerCase();
    var at = -1;
    terms.forEach(function (term) {
      var found = lower.indexOf(term);
      if (found !== -1 && (at === -1 || found < at)) at = found;
    });
    var start = Math.max(0, at - 50);
    var piece = text.slice(start, start + 170);
    var html = escapeHtml(piece);
    terms.forEach(function (term) {
      var safe = escapeHtml(term).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      html = html.replace(new RegExp(safe, "gi"), function (match) {
        return "<mark>" + match + "</mark>";
      });
    });
    return (start ? "…" : "") + html + (start + 170 < text.length ? "…" : "");
  }

  function runSearch() {
    var query = input.value.trim().toLowerCase();
    selected = -1;
    results.innerHTML = "";
    if (!query || !window.SEARCH_INDEX) return;
    var terms = query.split(/\s+/);
    var hits = [];
    window.SEARCH_INDEX.forEach(function (entry) {
      var title = (entry.s + " " + entry.p).toLowerCase();
      var body = entry.x.toLowerCase();
      var score = 0;
      for (var i = 0; i < terms.length; i++) {
        var inTitle = title.indexOf(terms[i]) !== -1;
        if (!inTitle && body.indexOf(terms[i]) === -1) return;
        score += inTitle ? 10 : 1;
      }
      hits.push({ entry: entry, score: score });
    });
    hits.sort(function (a, b) {
      return b.score - a.score;
    });
    if (!hits.length) {
      results.innerHTML = '<li class="search-empty">No results for this search.</li>';
      return;
    }
    results.innerHTML = hits
      .slice(0, 25)
      .map(function (hit) {
        var entry = hit.entry;
        return (
          '<li><a href="' + escapeHtml(root + entry.u) + '">' +
          '<span class="result-page">' + escapeHtml(entry.g + " · " + entry.p) + "</span>" +
          '<span class="result-title">' + escapeHtml(entry.s) + "</span>" +
          '<span class="result-snippet">' + snippet(entry.x, terms) + "</span></a></li>"
        );
      })
      .join("");
  }

  function openSearch() {
    if (dialog.open) return;
    dialog.showModal();
    input.select();
    loadIndex();
  }

  function moveSelection(step) {
    var links = results.querySelectorAll("a");
    if (!links.length) return;
    if (selected >= 0) links[selected].classList.remove("is-selected");
    selected = (selected + step + links.length) % links.length;
    links[selected].classList.add("is-selected");
    links[selected].scrollIntoView({ block: "nearest" });
  }

  document.querySelectorAll(".search-open").forEach(function (button) {
    button.addEventListener("click", openSearch);
  });
  input.addEventListener("input", runSearch);
  input.addEventListener("keydown", function (event) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      moveSelection(event.key === "ArrowDown" ? 1 : -1);
    } else if (event.key === "Enter") {
      event.preventDefault();
      var links = results.querySelectorAll("a");
      var target = links[selected >= 0 ? selected : 0];
      if (target) location.href = target.href;
    }
  });
  dialog.addEventListener("click", function (event) {
    if (event.target === dialog) dialog.close();
    else if (event.target.closest(".search-results a")) dialog.close();
  });
  document.addEventListener("keydown", function (event) {
    var tag = event.target.tagName;
    var typing = tag === "INPUT" || tag === "TEXTAREA";
    if ((event.key === "/" && !typing) || (event.key === "k" && (event.metaKey || event.ctrlKey))) {
      event.preventDefault();
      openSearch();
    }
  });
})();
