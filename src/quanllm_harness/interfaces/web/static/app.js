const $ = (selector) => document.querySelector(selector);

const ui = {
  health: $("#health"),
  question: $("#question"),
  token: $("#token"),
  submit: $("#submit"),
  stop: $("#stop"),
  workspace: $("#workspace"),
  timeline: $("#timeline"),
  elapsed: $("#elapsed"),
  reasoning: $("#reasoning"),
  answer: $("#answer"),
  status: $("#run-status"),
  usage: $("#usage"),
};

let controller = null;
let elapsedTimer = null;
let runStartedAt = null;
let serverElapsedSeconds = 0;
const reasoningPanels = new Map();

function formatElapsed(totalSeconds) {
  const total = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  return `${String(hours).padStart(2, "0")}小时${String(minutes).padStart(2, "0")}分钟${String(seconds).padStart(2, "0")}秒`;
}

function currentElapsedSeconds() {
  if (runStartedAt === null) return serverElapsedSeconds;
  return serverElapsedSeconds + (performance.now() - runStartedAt) / 1000;
}

function renderElapsed() {
  const elapsed = currentElapsedSeconds();
  ui.elapsed.textContent = formatElapsed(elapsed);
  ui.elapsed.dateTime = `PT${Math.floor(elapsed)}S`;
}

function startElapsedTimer() {
  serverElapsedSeconds = 0;
  runStartedAt = performance.now();
  clearInterval(elapsedTimer);
  renderElapsed();
  elapsedTimer = setInterval(renderElapsed, 250);
}

function syncElapsed(data) {
  if (typeof data?.elapsed_seconds !== "number") return;
  serverElapsedSeconds = data.elapsed_seconds;
  runStartedAt = performance.now();
  renderElapsed();
}

function stopElapsedTimer(data) {
  syncElapsed(data);
  if (runStartedAt !== null && typeof data?.elapsed_seconds !== "number") {
    serverElapsedSeconds = currentElapsedSeconds();
  }
  runStartedAt = null;
  clearInterval(elapsedTimer);
  elapsedTimer = null;
  renderElapsed();
}

function setHealth(kind, text) {
  ui.health.className = `health ${kind}`;
  ui.health.querySelector("span").textContent = text;
}

async function checkHealth() {
  try {
    const response = await fetch("/healthz");
    const data = await response.json();
    setHealth(data.configured ? "ready" : "error", `${data.model} · ${data.status}`);
  } catch {
    setHealth("error", "服务不可用");
  }
}

function timeline(stage, kind = "active") {
  const item = document.createElement("li");
  item.className = kind;
  item.textContent = stage;
  ui.timeline.appendChild(item);
  ui.timeline.scrollTop = ui.timeline.scrollHeight;
}

function resetRun() {
  ui.workspace.classList.remove("hidden");
  ui.timeline.replaceChildren();
  ui.reasoning.replaceChildren();
  reasoningPanels.clear();
  ui.answer.textContent = "等待终稿……";
  ui.usage.textContent = "";
  ui.status.className = "status-pill";
  ui.status.textContent = "运行中";
  startElapsedTimer();
}

function reasoningPanel(stage) {
  let panel = reasoningPanels.get(stage);
  if (panel) return panel;

  const section = document.createElement("section");
  section.className = "reasoning-stage";
  section.dataset.stage = stage;
  const heading = document.createElement("h3");
  heading.textContent = stage;
  const stream = document.createElement("pre");
  section.append(heading, stream);
  ui.reasoning.appendChild(section);
  panel = { section, stream };
  reasoningPanels.set(stage, panel);
  return panel;
}

function handleHarnessEvent(data) {
  syncElapsed(data);
  const { kind, stage, payload } = data;
  if (kind === "reasoning_delta") {
    const panel = reasoningPanel(stage);
    panel.stream.textContent += payload.text || "";
    panel.stream.scrollTop = panel.stream.scrollHeight;
    ui.reasoning.scrollTop = ui.reasoning.scrollHeight;
  } else if (kind === "agent_finished" || kind === "provider_error") {
    const panel = reasoningPanels.get(stage);
    if (panel) panel.section.classList.add("complete");
  } else if (kind === "agent_started") {
    const panel = reasoningPanels.get(stage);
    if (panel) {
      panel.section.classList.remove("complete");
      panel.stream.textContent += "\n\n──────── 新一轮 ────────\n";
    }
    timeline(stage);
  } else if (kind === "tool_finished") {
    timeline(`工具 ${stage} · ${payload.ok ? "成功" : "失败"}`, payload.ok ? "active" : "warning");
  } else if (kind === "repair_started") {
    timeline(`第 ${payload.round} 轮定向修复`, "warning");
  } else if (kind === "degraded") {
    timeline(`${stage} · 降级继续`, "warning");
  }
}

function finish(result, streamData = null) {
  stopElapsedTimer(streamData);
  ui.answer.textContent = result.answer || "未产生答案";
  ui.status.textContent = result.status;
  ui.status.className = `status-pill ${result.status}`;
  ui.usage.textContent = `修复 ${result.repair_rounds} 轮 · 输入 ${result.usage.prompt_tokens} Token · 输出 ${result.usage.completion_tokens} Token`;
}

function consumeBlock(block) {
  if (!block.trim() || block.startsWith(":")) return;
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  if (!data.length) return;
  const payload = JSON.parse(data.join("\n"));
  if (event === "request") syncElapsed(payload);
  if (event === "harness_event") handleHarnessEvent(payload);
  if (event === "result") finish(payload.result, payload);
  if (event === "error") throw new Error(payload.message || "请求失败");
  if (event === "cancelled") {
    stopElapsedTimer(payload);
    timeline("请求已取消", "warning");
  }
}

async function submit() {
  const question = ui.question.value.trim();
  if (!question || controller) return;
  resetRun();
  controller = new AbortController();
  ui.submit.disabled = true;
  ui.stop.disabled = false;
  const headers = { "Content-Type": "application/json", Accept: "text/event-stream" };
  if (ui.token.value) headers.Authorization = `Bearer ${ui.token.value}`;
  try {
    const response = await fetch("/api/v1/answers/stream", {
      method: "POST",
      headers,
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`${response.status}: ${detail}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      blocks.forEach(consumeBlock);
      if (done) {
        if (buffer.trim()) consumeBlock(buffer);
        break;
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      stopElapsedTimer();
      timeline("已在浏览器端停止", "warning");
    } else {
      stopElapsedTimer();
      ui.status.textContent = "请求失败";
      ui.status.className = "status-pill failed_without_answer";
      ui.answer.textContent = error.message;
    }
  } finally {
    controller = null;
    ui.submit.disabled = false;
    ui.stop.disabled = true;
  }
}

ui.submit.addEventListener("click", submit);
ui.stop.addEventListener("click", () => controller?.abort());
ui.question.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") submit();
});

checkHealth();
