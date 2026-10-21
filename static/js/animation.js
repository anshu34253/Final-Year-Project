/* ==========================================================================
   ViralShort AI - Animation & Scroll Engine JavaScript
   Scroll Reveal, Number Counter Animations & Real-time Processing Poller
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  initScrollObserver();
  initStatCounters();
});

/* 1. Intersection Observer for Scroll Reveals */
function initScrollObserver() {
  const revealElements = document.querySelectorAll('.reveal-on-scroll, .reveal-scale, .timeline-card, .feature-card, .tech-card');

  if (!('IntersectionObserver' in window)) {
    revealElements.forEach(el => el.classList.add('is-visible'));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
      }
    });
  }, {
    threshold: 0.15,
    rootMargin: '0px 0px -40px 0px'
  });

  revealElements.forEach(el => observer.observe(el));
}

/* 2. Number Counter Animation for Statistics Section */
function initStatCounters() {
  const statNumbers = document.querySelectorAll('.stat-number');
  if (!statNumbers.length) return;

  let animated = false;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !animated) {
        animated = true;
        statNumbers.forEach(counter => {
          const target = parseInt(counter.getAttribute('data-target') || '100', 10);
          const suffix = counter.getAttribute('data-suffix') || '';
          let count = 0;
          const duration = 2000; // 2s
          const increment = Math.ceil(target / (duration / 16));

          const updateCounter = () => {
            count += increment;
            if (count >= target) {
              counter.innerText = target + suffix;
            } else {
              counter.innerText = count + suffix;
              requestAnimationFrame(updateCounter);
            }
          };
          updateCounter();
        });
      }
    });
  }, { threshold: 0.5 });

  const statsSection = document.querySelector('.stats-section');
  if (statsSection) observer.observe(statsSection);
}
