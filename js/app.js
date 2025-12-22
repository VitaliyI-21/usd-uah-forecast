// web/js/app.js
let model = null;

async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Fetch failed: ${path} (${res.status})`);
  return await res.json();
}

// MinMaxScaler:
// scaled = x * scale + min_
// inverse: x = (scaled - min_) / scale
function norm(x, sc) { return x * sc.scale + sc.min_; }
function denorm(xScaled, sc) { return (xScaled - sc.min_) / sc.scale; }

function firstTensor(out) {
  if (!out) return null;
  if (Array.isArray(out)) return out[0];
  if (out.dataSync || out.data) return out; // Tensor
  if (typeof out === "object") {
    const k = Object.keys(out)[0];
    return out[k];
  }
  return null;
}

function findButtonByText(text) {
  const btns = Array.from(document.querySelectorAll("button"));
  return btns.find(b => (b.textContent || "").trim().toLowerCase() === text.toLowerCase()) || null;
}

function findValuesFallback() {
  // На твоєму UI є 3 “поля” підписані: Останній курс / Прогноз / Різниця
  // Вони найчастіше мають клас .value або схожу структуру.
  // Спробуємо знайти їх як три “бокси” зі значеннями.
  const candidates =
    Array.from(document.querySelectorAll(".value"))
    .filter(el => el && el.nodeType === 1);

  if (candidates.length >= 3) {
    return { lastEl: candidates[0], predEl: candidates[1], diffEl: candidates[2] };
  }

  // fallback: шукаємо 3 input/ div всередині карточки (якщо .value немає)
  const card = document.querySelector(".card") || document.body;
  const inputs = Array.from(card.querySelectorAll("input, .field, .output, div"))
    .filter(el => el && el.nodeType === 1);

  // дуже грубий fallback — краще щоб .value існували
  return { lastEl: candidates[0] || null, predEl: candidates[1] || null, diffEl: candidates[2] || null };
}

function findLookbackInput() {
  // якщо input lookback є — ок; якщо нема — беремо зі scaler
  return document.getElementById("lookback") || null;
}

async function main() {
  await tf.ready();

  // 1) знайти кнопки (по id або по тексту)
  const btnLoad =
    document.getElementById("btnLoad") ||
    document.getElementById("load") ||
    findButtonByText("Завантажити модель");

  const btnPredict =
    document.getElementById("btnPredict") ||
    document.getElementById("pred") ||
    findButtonByText("Прогноз");

  if (!btnLoad || !btnPredict) {
    throw new Error("Не знайдено кнопки. Перевір, що є кнопки 'Завантажити модель' та 'Прогноз'.");
  }

  const lookbackEl = findLookbackInput();

  // 2) знайти поля значень
  // спочатку пробуємо по id, якщо є
  let lastEl =
    document.getElementById("lastVal") ||
    document.getElementById("last") ||
    document.getElementById("lastRate") ||
    null;

  let predEl =
    document.getElementById("predVal") ||
    document.getElementById("forecast") ||
    document.getElementById("forecastRate") ||
    null;

  let diffEl =
    document.getElementById("diffVal") ||
    document.getElementById("diff") ||
    document.getElementById("diffRate") ||
    null;

  // якщо по id не знайшли — беремо 3 value-блоки по порядку
  if (!lastEl || !predEl || !diffEl) {
    const fb = findValuesFallback();
    lastEl = lastEl || fb.lastEl;
    predEl = predEl || fb.predEl;
    diffEl = diffEl || fb.diffEl;
  }

  if (!lastEl || !predEl || !diffEl) {
    throw new Error("Не знайдено поля для виводу (Останній курс/Прогноз/Різниця). Додай їм id або клас .value.");
  }

  // 3) підвантажити дані + scaler
  const demo = await loadJSON("./data/demo.json");
  const values = demo.values;
  const sc = demo.scaler;

  // заповнити last (щоб не було “—”)
  lastEl.textContent = Number(values[values.length - 1]).toFixed(4);

  // якщо lookback input існує — проставимо значення зі scaler
  if (lookbackEl && sc.lookback != null) lookbackEl.value = sc.lookback;

  // 4) load model
  btnLoad.addEventListener("click", async () => {
    try {
      btnLoad.disabled = true;
      btnLoad.textContent = "Завантаження...";
      model = await tf.loadGraphModel("./model/model.json");
      btnLoad.textContent = "Модель завантажено";
      btnPredict.disabled = false;

      console.log("inputs:", model.inputs?.map(i => i.name));
      console.log("outputs:", model.outputs?.map(o => o.name));
    } catch (e) {
      console.error(e);
      alert("Помилка завантаження моделі: " + e.message);
      btnLoad.disabled = false;
      btnLoad.textContent = "Завантажити модель";
    }
  });

  // 5) predict (IMPORTANT: executeAsync)
  btnPredict.addEventListener("click", async () => {
    try {
      if (!model) { alert("Спочатку завантаж модель."); return; }

      const lookback =
        lookbackEl ? parseInt(lookbackEl.value, 10)
                   : (sc.lookback != null ? parseInt(sc.lookback, 10) : 5);

      if (!Number.isFinite(lookback) || lookback < 1) {
        throw new Error("Невірний lookback.");
      }
      if (values.length < lookback) {
        throw new Error(`Недостатньо даних: треба ${lookback}, є ${values.length}`);
      }

      const windowRaw = values.slice(values.length - lookback);
      const windowScaled = windowRaw.map(v => norm(Number(v), sc));
      const x = tf.tensor(windowScaled, [1, lookback, 1], "float32");

      // ✅ LSTM GraphModel → тільки executeAsync
      const out = await model.executeAsync(x);
      const yTensor = firstTensor(out);
      if (!yTensor) throw new Error("Модель не повернула Tensor.");

      const yVal = (await yTensor.data())[0];
      const forecast = denorm(yVal, sc);

      const last = Number(windowRaw[windowRaw.length - 1]);
      const diff = forecast - last;

      predEl.textContent = Number(forecast).toFixed(4);
      diffEl.textContent = (diff >= 0 ? "+" : "") + Number(diff).toFixed(4);

      tf.dispose(x);
      if (Array.isArray(out)) out.forEach(t => t.dispose && t.dispose());
      else if (out?.dispose) out.dispose();
    } catch (e) {
      console.error(e);
      alert("Помилка прогнозу: " + e.message);
    }
  });
}

// гарантуємо, що DOM готовий
window.addEventListener("DOMContentLoaded", () => {
  main().catch(err => {
    console.error(err);
    alert("Помилка: " + err.message);
  });
});
