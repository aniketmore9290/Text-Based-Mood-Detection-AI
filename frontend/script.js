const API_URL = "http://localhost:8000/predict";

const MOOD_METADATA = {
  Happy: {
    accentColor: "#10b981",
    barColor: "linear-gradient(90deg, #10b981, #f59e0b)",
    emoji: "😊",
    cardClass: "mood-happy",
    tagline: "Joyful, energized, or celebratory sentiment detected.",
  },
  Sad: {
    accentColor: "#3b82f6",
    barColor: "linear-gradient(90deg, #3b82f6, #60a5fa)",
    emoji: "😢",
    cardClass: "mood-sad",
    tagline: "Sorrow, melancholy, or disheartened sentiment detected.",
  },
  Angry: {
    accentColor: "#ef4444",
    barColor: "linear-gradient(90deg, #ef4444, #f87171)",
    emoji: "😠",
    cardClass: "mood-angry",
    tagline: "Frustration, indignation, or irritation detected.",
  },
  Neutral: {
    accentColor: "#94a3b8",
    barColor: "linear-gradient(90deg, #64748b, #94a3b8)",
    emoji: "😐",
    cardClass: "mood-neutral",
    tagline: "Informational, calm, or objective statement detected.",
  },
};

const moodInput = document.getElementById("moodInput");
const clearBtn = document.getElementById("clearBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnText = document.getElementById("btnText");
const errorBanner = document.getElementById("errorBanner");
const errorText = document.getElementById("errorText");
const resultCard = document.getElementById("resultCard");
const resultEmoji = document.getElementById("resultEmoji");
const resultMoodName = document.getElementById("resultMoodName");
const resultTagline = document.getElementById("resultTagline");
const resultConfidence = document.getElementById("resultConfidence");
const confidenceBarFill = document.getElementById("confidenceBarFill");
const probabilitiesSection = document.getElementById("probabilitiesSection");
const probabilitiesGrid = document.getElementById("probabilitiesGrid");
const sampleChips = document.querySelectorAll(".chip");

function updateClearButton() {
  clearBtn.style.display = moodInput.value ? "flex" : "none";
}

moodInput.addEventListener("input", updateClearButton);

clearBtn.addEventListener("click", () => {
  moodInput.value = "";
  updateClearButton();
  moodInput.focus();
});

moodInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    analyzeMood();
  }
});

analyzeBtn.addEventListener("click", () => analyzeMood());

sampleChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    const text = chip.getAttribute("data-text");
    moodInput.value = text;
    updateClearButton();
    analyzeMood(text);
  });
});

async function analyzeMood(textOverride) {
  const query = (textOverride !== undefined ? textOverride : moodInput.value).trim();
  if (!query) return;

  setLoading(true);
  hideError();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: query }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server responded with status ${response.status}`);
    }

    const data = await response.json();
    displayResult(data);
  } catch (err) {
    showError(
      err.message.includes("Failed to fetch")
        ? "Unable to reach the backend at http://localhost:8000. Please ensure FastAPI is running."
        : err.message
    );
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  if (isLoading) {
    btnText.innerHTML = '<span class="spinner"></span> Analyzing Mood...';
  } else {
    btnText.innerHTML = "Analyze Mood ➔";
  }
}

function showError(message) {
  errorText.textContent = message;
  errorBanner.style.display = "flex";
  resultCard.style.display = "none";
}

function hideError() {
  errorBanner.style.display = "none";
}

function displayResult(data) {
  if (data.flagged || data.mood === "Flagged") {
    resultCard.className = "result-card mood-flagged";
    resultEmoji.textContent = data.emoji || "⚠️";
    resultMoodName.textContent = "Flagged Content";
    resultTagline.textContent =
      data.warning || "Inappropriate or harmful intent detected. Please enter respectful text.";
    resultConfidence.textContent = "100%";

    confidenceBarFill.style.width = "100%";
    confidenceBarFill.style.background = "linear-gradient(90deg, #f59e0b, #ef4444)";

    if (probabilitiesSection) {
      probabilitiesSection.style.display = "none";
    }
    resultCard.style.display = "flex";
    return;
  }

  if (probabilitiesSection) {
    probabilitiesSection.style.display = "flex";
  }

  const meta = MOOD_METADATA[data.mood] || MOOD_METADATA.Neutral;

  resultCard.className = `result-card ${meta.cardClass}`;
  resultEmoji.textContent = data.emoji;
  resultMoodName.textContent = data.mood;
  resultTagline.textContent = meta.tagline;
  resultConfidence.textContent = `${data.confidence}%`;

  confidenceBarFill.style.width = `${data.confidence}%`;
  confidenceBarFill.style.background = meta.barColor;

  probabilitiesGrid.innerHTML = "";
  const probabilities = data.probabilities || {};

  Object.entries(probabilities).forEach(([moodKey, prob]) => {
    const percent = (prob * 100).toFixed(1);
    const isDetected = moodKey === data.mood;
    const moodMeta = MOOD_METADATA[moodKey] || {};

    const item = document.createElement("div");
    item.className = `probability-item ${isDetected ? "active-item" : ""}`;

    item.innerHTML = `
      <div class="prob-header">
        <span class="prob-mood-name">
          <span class="prob-emoji">${moodMeta.emoji || "•"}</span>
          ${moodKey}
        </span>
        <span class="prob-percentage">${percent}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${percent}%; background-color: ${moodMeta.accentColor || '#94a3b8'};"></div>
      </div>
    `;

    probabilitiesGrid.appendChild(item);
  });

  resultCard.style.display = "flex";
}
