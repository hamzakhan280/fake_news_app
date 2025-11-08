// Wait until DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {

  // Animate confidence bars smoothly
  const animateBars = () => {
    const bars = document.querySelectorAll(".bar-fill");
    bars.forEach(bar => {
      const targetWidth = bar.style.width;
      bar.style.width = "0%"; // start from 0
      setTimeout(() => {
        bar.style.transition = "width 1.2s ease-out";
        bar.style.width = targetWidth;
      }, 200);
    });
  };

  // Fade-in animation for the result card
  const fadeInCard = () => {
    const card = document.querySelector(".result-card");
    if (card) {
      card.style.opacity = 0;
      card.style.transform = "translateY(15px)";
      setTimeout(() => {
        card.style.transition = "opacity 0.8s ease, transform 0.8s ease";
        card.style.opacity = 1;
        card.style.transform = "translateY(0)";
      }, 150);
    }
  };

  // Animate the percentage counters (ML & AI confidence)
  const animateValues = () => {
    const values = document.querySelectorAll(".bar-value");
    values.forEach(valueEl => {
      const targetValue = parseInt(valueEl.textContent);
      let current = 0;
      const increment = Math.ceil(targetValue / 40);
      const updateValue = () => {
        current += increment;
        if (current >= targetValue) {
          valueEl.textContent = targetValue + "%";
        } else {
          valueEl.textContent = current + "%";
          requestAnimationFrame(updateValue);
        }
      };
      updateValue();
    });
  };

  // Run all animations if on the result page
  if (document.querySelector(".result-card")) {
    animateBars();
    fadeInCard();
    animateValues();
  }

  // Smooth scroll to the top when clicking "New Check"
  const newCheckBtn = document.querySelector(".btn.secondary");
  if (newCheckBtn) {
    newCheckBtn.addEventListener("click", (e) => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

});
