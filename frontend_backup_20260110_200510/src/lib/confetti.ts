export function triggerConfetti() {
  // Simple confetti effect using CSS animations
  const colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff"];
  const container = document.createElement("div");
  container.style.position = "fixed";
  container.style.top = "0";
  container.style.left = "0";
  container.style.width = "100%";
  container.style.height = "100%";
  container.style.pointerEvents = "none";
  container.style.zIndex = "9999";
  
  for (let i = 0; i < 50; i++) {
    const confetti = document.createElement("div");
    confetti.style.position = "absolute";
    confetti.style.width = "10px";
    confetti.style.height = "10px";
    confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
    confetti.style.left = Math.random() * 100 + "%";
    confetti.style.top = "-10px";
    confetti.style.borderRadius = "50%";
    confetti.style.animation = `fall ${2 + Math.random() * 2}s linear forwards`;
    container.appendChild(confetti);
  }
  
  document.body.appendChild(container);
  
  setTimeout(() => {
    container.remove();
  }, 4000);
}

// Add CSS animation
const style = document.createElement("style");
style.textContent = `
  @keyframes fall {
    to {
      transform: translateY(100vh) rotate(720deg);
      opacity: 0;
    }
  }
`;
document.head.appendChild(style);

export default triggerConfetti;
