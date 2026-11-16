/**
 * app.js — Frontend Controller for EduLens
 * Refined for clarity, professional presentation, and enhanced UX.
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
  const copySchemaBtn = document.getElementById("copy-schema-btn");
  const copyBtnText = document.getElementById("copy-btn-text");
  const errorAlert = document.getElementById("error-alert");
  const errorMessage = document.getElementById("error-message");
  const loadingState = document.getElementById("loading-state");
  const resultsSection = document.getElementById("results-section");
  const uploadSection = document.getElementById("upload-section");
  const analyzeAnotherBtn = document.getElementById("analyze-another-btn");

  const navAnalyzeBtn = document.getElementById("nav-analyze-btn");
  const navAboutBtn = document.getElementById("nav-about-btn");
  const aboutSection = document.getElementById("about-section");

  const statusText = document.getElementById("status-text");
  const statusDot = document.getElementById("status-dot");
  const statusPulse = document.getElementById("status-pulse");

  // State
  let currentFile = null;
  let inMemorySample = null;

  // Check Backend Status
  checkSystemStatus();

  async function checkSystemStatus() {
    try {
      const resp = await fetch("/api/health");
      if (resp.ok) {
        const data = await resp.json();
        if (data.model_loaded) {
          statusText.textContent = "Learning Analysis • Ready";
          statusDot.className = "relative inline-flex rounded-full h-2 w-2 bg-emerald-500";
          statusPulse.className = "animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75";
          return;
        }
      }
      throw new Error();
    } catch {
      statusText.textContent = "System Connecting...";
      statusDot.className = "relative inline-flex rounded-full h-2 w-2 bg-amber-500";
      statusPulse.className = "animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-500 opacity-75";
    }
  }

  // Navigation Scrolling
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

  // Copy Schema Button
  if (copySchemaBtn) {
    copySchemaBtn.addEventListener("click", () => {
      const sampleJSON = JSON.stringify({
        topic: "Binary Search",
        messages: [
          { role: "assistant", content: "How does binary search work?" },
          { role: "student", content: "It repeatedly divides the search space in half!" }
        ]
      }, null, 2);

      navigator.clipboard.writeText(sampleJSON).then(() => {
        copyBtnText.textContent = "Copied!";
        setTimeout(() => {
          copyBtnText.textContent = "Copy";
        }, 2000);
      });
    });
  }

  // File Dropzone Interaction
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

  // Clear File Selection
  clearFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetFileInput();
  });

  function setFile(file) {
    if (!file.name.toLowerCase().endsWith(".json")) {
      showError("Please upload a .json file. Other file formats are not supported.");
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
      if (!resp.ok) throw new Error("Could not load sample conversation.");
      const sample = await resp.json();
      inMemorySample = sample;
      currentFile = null;
      fileInput.value = "";

      topicInput.value = sample.topic || "Physics: Static vs. Kinetic Friction";
      fileNameDisplay.textContent = "sample_friction_concept.json (Sample Loaded)";
      fileInfoBadge.classList.remove("hidden");
    } catch (err) {
      showError(`Unable to load sample conversation: ${err.message}`);
    }
  });

  // Analyze Action
  analyzeBtn.addEventListener("click", async () => {
    hideError();

    if (!currentFile && !inMemorySample) {
      showError("Please select a JSON conversation file or load a sample conversation first.");
      return;
    }

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

      renderResults(data);

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
    // 1. Topic name and status
    document.getElementById("results-topic-badge").textContent = `Topic: ${data.topic || "General"}`;

    // 2. Student Understanding Result
    const und = data.understanding || {};
    const undLabel = und.label || "Understood";
    const undProb = Math.round((und.probability || 0.8) * 100);

    const badgeEl = document.getElementById("understanding-badge");
    const labelEl = document.getElementById("understanding-label");
    const probRingEl = document.getElementById("prob-ring-fill");
    const probValEl = document.getElementById("understanding-prob-val");
    const probRingValEl = document.getElementById("understanding-prob-val-ring");
    const explEl = document.getElementById("understanding-explanation-text");

    labelEl.textContent = undLabel;
    probValEl.textContent = `${undProb}%`;
    probRingValEl.textContent = `${undProb}%`;
    explEl.textContent = und.explanation || "";

    probRingEl.setAttribute("stroke-dasharray", `${undProb}, 100`);

    if (undLabel === "Understood") {
      badgeEl.className = "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full font-display font-bold text-base bg-emerald-50 text-emerald-700 border border-emerald-200";
      badgeEl.firstElementChild.className = "w-2.5 h-2.5 rounded-full bg-emerald-600";
      probRingEl.setAttribute("stroke", "#059669");
    } else if (undLabel === "Partially Understood") {
      badgeEl.className = "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full font-display font-bold text-base bg-amber-50 text-amber-700 border border-amber-200";
      badgeEl.firstElementChild.className = "w-2.5 h-2.5 rounded-full bg-amber-600";
      probRingEl.setAttribute("stroke", "#d97706");
    } else {
      badgeEl.className = "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full font-display font-bold text-base bg-rose-50 text-rose-700 border border-rose-200";
      badgeEl.firstElementChild.className = "w-2.5 h-2.5 rounded-full bg-rose-600";
      probRingEl.setAttribute("stroke", "#e11d48");
    }

    // 3. Expressed Learning Confidence ("How Confident Did the Student Sound?")
    const conf = data.expressed_confidence || {};
    const confLabel = conf.label || "Moderate Confidence";
    const confScore = Math.round(conf.score || 50);

    const confBadgeEl = document.getElementById("confidence-badge");
    const confLabelEl = document.getElementById("confidence-label");
    const confScoreEl = document.getElementById("confidence-score-val");
    const confBarEl = document.getElementById("confidence-progress-bar");
    const confDescEl = document.getElementById("confidence-plain-description");

    confLabelEl.textContent = confLabel;
    confScoreEl.textContent = `${confScore}%`;
    confBarEl.style.width = `${Math.min(100, Math.max(5, confScore))}%`;

    // Dynamic Plain-Language Explanation
    if (confLabel.includes("High")) {
      confBadgeEl.className = "inline-flex items-center gap-1.5 px-3 py-1 rounded-full font-display font-bold text-sm bg-emerald-50 text-emerald-700 border border-emerald-200";
      confBarEl.className = "h-full bg-success rounded-full transition-all duration-500";
      confDescEl.textContent = "The student generally expressed certainty, explained ideas clearly, and showed fewer signs of hesitation.";
    } else if (confLabel.includes("Moderate")) {
      confBadgeEl.className = "inline-flex items-center gap-1.5 px-3 py-1 rounded-full font-display font-bold text-sm bg-amber-50 text-amber-700 border border-amber-200";
      confBarEl.className = "h-full bg-warning rounded-full transition-all duration-500";
      confDescEl.textContent = "The student showed some certainty but also expressed uncertainty or needed clarification at certain points.";
    } else {
      confBadgeEl.className = "inline-flex items-center gap-1.5 px-3 py-1 rounded-full font-display font-bold text-sm bg-rose-50 text-rose-700 border border-rose-200";
      confBarEl.className = "h-full bg-danger rounded-full transition-all duration-500";
      confDescEl.textContent = "The student frequently expressed doubt, hesitation, or difficulty explaining the topic.";
    }

    // "What influenced this estimate?" — Only actually detected signals
    const signalsContainer = document.getElementById("confidence-signals-container");
    signalsContainer.innerHTML = "";
    const sigs = conf.signals || {};
    let detectedSignalsCount = 0;

    if (sigs.certainty_count > 0) {
      detectedSignalsCount++;
      const chip = document.createElement("span");
      chip.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200";
      chip.innerHTML = `<span class="material-symbols-outlined text-[13px]">check_circle</span> ${Math.round(sigs.certainty_count)} expression(s) of certainty`;
      signalsContainer.appendChild(chip);
    }

    if (sigs.independent_explanations > 0) {
      detectedSignalsCount++;
      const chip = document.createElement("span");
      chip.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200";
      chip.innerHTML = `<span class="material-symbols-outlined text-[13px]">lightbulb</span> ${Math.round(sigs.independent_explanations)} independent explanation(s)`;
      signalsContainer.appendChild(chip);
    }

    if (sigs.self_corrections > 0) {
      detectedSignalsCount++;
      const chip = document.createElement("span");
      chip.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-blue-50 text-primary border border-blue-200";
      chip.innerHTML = `<span class="material-symbols-outlined text-[13px]">edit_note</span> ${Math.round(sigs.self_corrections)} self-correction(s)`;
      signalsContainer.appendChild(chip);
    }

    if (sigs.uncertainty_count > 0) {
      detectedSignalsCount++;
      const chip = document.createElement("span");
      chip.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-rose-50 text-rose-700 border border-rose-200";
      chip.innerHTML = `<span class="material-symbols-outlined text-[13px]">help</span> ${Math.round(sigs.uncertainty_count)} uncertainty hedge(s)`;
      signalsContainer.appendChild(chip);
    }

    if (sigs.hesitation_count > 0) {
      detectedSignalsCount++;
      const chip = document.createElement("span");
      chip.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-amber-50 text-amber-700 border border-amber-200";
      chip.innerHTML = `<span class="material-symbols-outlined text-[13px]">pause</span> ${Math.round(sigs.hesitation_count)} hesitation marker(s)`;
      signalsContainer.appendChild(chip);
    }

    if (sigs.confirmation_requests > 0) {
      detectedSignalsCount++;
      const chip = document.createElement("span");
      chip.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-amber-50 text-amber-700 border border-amber-200";
      chip.innerHTML = `<span class="material-symbols-outlined text-[13px]">help_outline</span> ${Math.round(sigs.confirmation_requests)} confirmation check(s)`;
      signalsContainer.appendChild(chip);
    }

    if (detectedSignalsCount === 0) {
      const chip = document.createElement("span");
      chip.className = "inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-100 text-darktext border border-bordercolor";
      chip.innerHTML = `<span class="material-symbols-outlined text-[13px]">verified</span> Zero hesitation or uncertainty phrases detected`;
      signalsContainer.appendChild(chip);
    }

    // 4. Conversation Statistics
    const stats = data.conversation_stats || {};
    document.getElementById("stat-student-turns").textContent = stats.student_messages || 0;
    document.getElementById("stat-assistant-turns").textContent = stats.assistant_messages || 0;
    document.getElementById("stat-total-words").textContent = stats.total_words || 0;
    document.getElementById("stat-question-count").textContent = stats.question_count || 0;

    // 5. Important Observations
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

    // 6. Class Probability Distribution
    const probs = und.class_probabilities || {};
    const pUnderstood = Math.round((probs["Understood"] || 0) * 100);
    const pPartial = Math.round((probs["Partially Understood"] || 0) * 100);
    const pConfused = Math.round((probs["Confused"] || 0) * 100);

    document.getElementById("prob-understood-text").textContent = `${pUnderstood}%`;
    document.getElementById("prob-partial-text").textContent = `${pPartial}%`;
    document.getElementById("prob-confused-text").textContent = `${pConfused}%`;
  }

  // Analyze Another Conversation
  analyzeAnotherBtn.addEventListener("click", () => {
    resultsSection.classList.add("hidden");
    resetFileInput();
    topicInput.value = "";
    uploadSection.scrollIntoView({ behavior: "smooth" });
  });
});
