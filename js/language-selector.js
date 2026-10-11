(function () {
  "use strict";

  var STORAGE_KEY = "moo-lang";
  var DEFAULT = "pymoors";

  function stored() {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT;
  }

  function save(lang) {
    localStorage.setItem(STORAGE_KEY, lang);
  }

  function applyLanguage(lang) {
    document.documentElement.setAttribute("data-lang", lang);
    var sel = document.querySelector(".moo-lang-select");
    if (sel) sel.value = lang;
  }

  function injectSelect() {
    if (document.querySelector(".moo-lang-select")) return;
    var inner = document.querySelector(".md-header__inner");
    if (!inner) return;

    var wrap = document.createElement("div");
    wrap.className = "moo-lang-select-wrap";
    wrap.innerHTML =
      '<select class="moo-lang-select" aria-label="Select documentation language">' +
      '<option value="pymoors">pymoors</option>' +
      '<option value="moors">moors</option>' +
      '</select>';

    inner.appendChild(wrap);

    wrap.querySelector("select").addEventListener("change", function (e) {
      var lang = e.target.value;
      save(lang);
      applyLanguage(lang);
    });
  }

  // Hero "pymoors docs / moors docs" links set language before navigating
  document.addEventListener("click", function (e) {
    var link = e.target.closest("[data-setlang]");
    if (!link) return;
    var lang = link.dataset.setlang;
    save(lang);
    applyLanguage(lang);
  });

  function onPageReady() {
    injectSelect();
    applyLanguage(stored());
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(onPageReady);
  } else {
    document.addEventListener("DOMContentLoaded", onPageReady);
  }
})();
