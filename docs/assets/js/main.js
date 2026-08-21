/* Vanilla, dependency-free. Two jobs: the BibTeX copy button and nav scrollspy. */

/* --- BibTeX copy ------------------------------------------------------- */
const copyBtn = document.getElementById("copy-bibtex");

copyBtn?.addEventListener("click", async () => {
  const text = document.getElementById("bibtex-text").innerText;
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = "Copied";
    copyBtn.classList.add("is-copied");
  } catch (err) {
    copyBtn.textContent = "Copy failed";
  }
  setTimeout(() => {
    copyBtn.textContent = "Copy";
    copyBtn.classList.remove("is-copied");
  }, 1800);
});

/* --- Nav scrollspy ----------------------------------------------------- */
const navLinks = Array.from(document.querySelectorAll(".site-nav .nav-link"));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

if (sections.length && "IntersectionObserver" in window) {
  const visible = new Set();

  const setActive = () => {
    // Topmost visible section wins, so the marker never lags behind the reader.
    const current = sections.find((section) => visible.has(section));
    navLinks.forEach((link) => {
      const isActive = Boolean(current) && link.getAttribute("href") === `#${current.id}`;
      link.classList.toggle("is-active", isActive);
    });
  };

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) visible.add(entry.target);
        else visible.delete(entry.target);
      });
      setActive();
    },
    // Discount the sticky nav at the top and most of the viewport at the
    // bottom, so a section counts as "current" once it reaches the top third.
    { rootMargin: "-56px 0px -65% 0px", threshold: 0 }
  );

  sections.forEach((section) => observer.observe(section));
}
