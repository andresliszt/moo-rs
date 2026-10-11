document$.subscribe(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.startup.output.clearCache();
        window.MathJax.typesetPromise();
    }
});
