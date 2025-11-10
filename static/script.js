document.addEventListener("DOMContentLoaded", () => {
  // Animate bar fills smoothly
  const fills = document.querySelectorAll(".bar-fill");
  fills.forEach((fill) => {
    const target = parseFloat(fill.dataset.value || fill.style.width);
    fill.style.width = "0%";
    setTimeout(() => {
      fill.style.transition = "width 1.5s ease-out";
      fill.style.width = target + "%";
    }, 200);
  });

  // Animate value numbers
  const values = document.querySelectorAll(".score-value");
  values.forEach((el) => {
    const final = parseFloat(el.textContent) || 0;
    let cur = 0;
    const step = final / 60;
    const interval = setInterval(() => {
      cur += step;
      if (cur >= final) {
        el.textContent = final.toFixed(1) + "%";
        clearInterval(interval);
      } else {
        el.textContent = cur.toFixed(1) + "%";
      }
    }, 25);
  });

  // Scroll to result card
  const resultCard = document.querySelector(".result-card");
  if (resultCard) {
    setTimeout(() => {
      window.scrollTo({
        top: resultCard.offsetTop - 40,
        behavior: "smooth",
      });
    }, 300);
  }

  // Theme toggle (optional)
  const toggle = document.createElement("button");
  toggle.className = "theme-toggle";
  toggle.innerHTML = "🌓";
  document.body.appendChild(toggle);

  toggle.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    toggle.classList.toggle("active");
  });
});
