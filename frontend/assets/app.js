/**
 * app.js — Frontend Controller for EduLens Mini
 * Handles file drop/selection, API requests to /api/analyze, dynamic metric rendering,
 * and user interactions.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("json-file-input");
  const fileInfoBadge = document.getElementById("file-info-badge");
  const fileNameDisplay = document.getElementById("file-name-display");
  const clearFileBtn = document.getElementById("clear-file-btn");
  const topicInput = document.getElementById("topic-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const loadSampleBtn = document.getElementById("load-sample-btn");
  const errorAlert = document.getElementById("error-alert");
  const errorMessage = document.getElementById("error-message");
  const loadingState = document.getElementById("loading-state");
  const resultsSection = document.getElementById("results-section");
  const uploadSection = document.getElementById("upload-section");
  const analyzeAnotherBtn = document.getElementById("analyze-another-btn");

  const navAnalyzeBtn = document.getElementById("nav-analyze-btn");
  const navAboutBtn = document.getElementById("nav-about-btn");
  const aboutSection = document.getElementById("about-section");

  // State
  let currentFile = null;
  let inMemorySample = null;

  // Navigation
  if (navAnalyzeBtn) {
    navAnalyzeBtn.addEventListener("click", () => {
      uploadSection.scrollIntoView({ behavior: "smooth" });
    });
  }
  if (navAboutBtn) {
    navAboutBtn.addEventListener("click", () => {
      aboutSection.scrollIntoView({ behavior: "smooth" });
    });
  }

  // File Input and Dropzone
  dropzone.addEventListener("click", (e) => {
    if (e.target !== clearFileBtn && !clearFileBtn.contains(e.target)) {
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  });

  // Drag & Drop
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dropzone-active");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dropzone-active");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      setFile(dt.files[0]);
    }
  });

  // Clear File
  clearFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetFileInput();
  });

  function setFile(file) {
    if (!file.name.toLowerCase().endsWith(".json")) {
      showError("Please upload a .json file. Other file extensions are not supported.");
      return;
    }
    currentFile = file;
    inMemorySample = null;
    fileNameDisplay.textContent = `${file.name} (${formatBytes(file.size)})`;
    fileInfoBadge.classList.remove("hidden");
    hideError();
  }

  function resetFileInput() {
    currentFile = null;
    inMemorySample = null;
    fileInput.value = "";
    fileInfoBadge.classList.add("hidden");
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.classList.remove("hidden");
  }

  function hideError() {
    errorAlert.classList.add("hidden");
  }

  // Load Sample Conversation
  loadSampleBtn.addEventListener("click", async () => {
    hideError();
    try {
      const resp = await fetch("/api/sample-conversation");
      if (!resp.ok) throw new Error("Failed to load sample conversation.");
      const sample = await resp.json();
      inMemorySample = sample;
      currentFile = null;
      fileInput.value = "";

      topicInput.value = sample.topic || "Physics: Static vs. Kinetic Friction";
      fileNameDisplay.textContent = "sample_physics_friction.json (Pre-loaded Sample)";
      fileInfoBadge.classList.remove("hidden");
    } catch (err) {
      showError(`Unable to load sample conversation: ${err.message}`);
    }
  });

  // Analyze Button Action
  analyzeBtn.addEventListener("click", async () => {
    hideError();

    if (!currentFile && !inMemorySample) {
      showError("Please select a JSON conversation file or load a sample conversation first.");
      return;
    }

    // Set Loading UI
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add("opacity-60", "cursor-not-allowed");
    loadingState.classList.remove("hidden");
    resultsSection.classList.add("hidden");

    try {
      let response;

      if (currentFile) {
        const formData = new FormData();
        formData.append("file", currentFile);
        if (topicInput.value.trim()) {
          formData.append("topic", topicInput.value.trim());
        }

        response = await fetch("/api/analyze", {
          method: "POST",
          body: formData,
        });
      } else if (inMemorySample) {
        const payload = { ...inMemorySample };
        if (topicInput.value.trim()) {
          payload.topic = topicInput.value.trim();
        }

        response = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Analysis request failed.");
      }

      // Render Results
      renderResults(data);

      // Scroll to Results
      resultsSection.classList.remove("hidden");
      resultsSection.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
      showError(err.message || "An error occurred during analysis.");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove("opacity-60", "cursor-not-allowed");
      loadingState.classList.add("hidden");
    }
  });

  // Render Analysis Results
  function renderResults(data) {
    // Topic
    document.getElementById("results-topic-badge").textContent = `Topic: ${data.topic || "General"}`;

    // 1. Understanding
    const und = data.understanding || {};
    const undLabel = und.label || "Understood";
    const undProb = Math.round((und.probability || 0.8) * 100);

    const badgeEl = document.getElementById("understanding-badge");
    const labelEl = document.getElementById("understanding-label");
    const probRingEl = document.getElementById("prob-ring-fill");
    const probValEl = document.getElementById("understanding-prob-val");
    const explEl = document.getElementById("understanding-explanation-text");

    labelEl.textContent = undLabel;
    probValEl.textContent = `${undProb}%`;
    explEl.textContent = und.explanation || "";

    // Probability Ring SVG dasharray
    probRingEl.setAttribute("stroke-dasharray", `${undProb}, 100`);

    // Badge and Ring Colors
    if (undLabel === "Understood") {
      badgeEl.className = "inline-flex items-center gap-2 px-4 py-1.5 rounded-full font-display font-bold text-base bg-emerald-50 text-emerald-700 border border-emerald-200";
      badgeEl.firstElementChild.className = "w-2.5 h-2.5 rounded-full bg-emerald-600";
      probRingEl.setAttribute("stroke", "#059669");
    } else if (undLabel === "Partially Understood") {
      badgeEl.className = "inline-flex items-center gap-2 px-4 py-1.5 rounded-full font-display font-bold text-base bg-amber-50 text-amber-700 border border-amber-200";
      badgeEl.firstElementChild.className = "w-2.5 h-2.5 rounded-full bg-amber-600";
      probRingEl.setAttribute("stroke", "#d97706");
    } else {
      badgeEl.className = "inline-flex items-center gap-2 px-4 py-1.5 rounded-full font-display font-bold text-base bg-rose-50 text-rose-700 border border-rose-200";
      badgeEl.firstElementChild.className = "w-2.5 h-2.5 rounded-full bg-rose-600";
      probRingEl.setAttribute("stroke", "#e11d48");
    }

    // Probability Bars
    const probs = und.class_probabilities || {};
    const pUnderstood = Math.round((probs["Understood"] || 0) * 100);
    const pPartial = Math.round((probs["Partially Understood"] || 0) * 100);
    const pConfused = Math.round((probs["Confused"] || 0) * 100);

    document.getElementById("prob-understood-text").textContent = `${pUnderstood}%`;
    document.getElementById("prob-understood-bar").style.width = `${pUnderstood}%`;

    document.getElementById("prob-partial-text").textContent = `${pPartial}%`;
    document.getElementById("prob-partial-bar").style.width = `${pPartial}%`;

    document.getElementById("prob-confused-text").textContent = `${pConfused}%`;
    document.getElementById("prob-confused-bar").style.width = `${pConfused}%`;

    // 2. Expressed Confidence
    const conf = data.expressed_confidence || {};
    const confLabel = conf.label || "Moderate Confidence";
    const confScore = Math.round(conf.score || 50);

    const confBadgeEl = document.getElementById("confidence-badge");
    const confLabelEl = document.getElementById("confidence-label");
    const confScoreEl = document.getElementById("confidence-score-val");

    confLabelEl.textContent = confLabel;
    confScoreEl.textContent = `${confScore}%`;

    if (confLabel.includes("High")) {
      confBadgeEl.className = "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full font-display font-bold text-sm bg-emerald-50 text-emerald-700 border border-emerald-200";
    } else if (confLabel.includes("Moderate")) {
      confBadgeEl.className = "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full font-display font-bold text-sm bg-amber-50 text-amber-700 border border-amber-200";
    } else {
      confBadgeEl.className = "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full font-display font-bold text-sm bg-rose-50 text-rose-700 border border-rose-200";
    }

    const sigs = conf.signals || {};
    document.getElementById("marker-certainty").textContent = Math.round(sigs.certainty_count || 0);
    document.getElementById("marker-uncertainty").textContent = Math.round(sigs.uncertainty_count || 0);
    document.getElementById("marker-hesitation").textContent = Math.round(sigs.hesitation_count || 0);
    document.getElementById("marker-selfcorr").textContent = Math.round(sigs.self_corrections || 0);

    // 3. Stats & Insights
    const stats = data.conversation_stats || {};
    document.getElementById("stat-student-turns").textContent = stats.student_messages || 0;
    document.getElementById("stat-assistant-turns").textContent = stats.assistant_messages || 0;
    document.getElementById("stat-total-words").textContent = stats.total_words || 0;
    document.getElementById("stat-question-count").textContent = stats.question_count || 0;

    const insightsListEl = document.getElementById("insights-list");
    insightsListEl.innerHTML = "";
    const insights = data.insights || [];
    if (insights.length === 0) {
      insights.push("Student completed discussion session on topic.");
    }
    insights.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      insightsListEl.appendChild(li);
    });
  }

  // Analyze Another Conversation
  analyzeAnotherBtn.addEventListener("click", () => {
    resultsSection.classList.add("hidden");
    resetFileInput();
    topicInput.value = "";
    uploadSection.scrollIntoView({ behavior: "smooth" });
  });
});
