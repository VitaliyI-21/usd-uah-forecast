let model = null;

async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Fetch failed: ${path} (${res.status})`);
  return await res.json();
}

// MinMaxScaler params:
// x_scaled = x * scale + min_
// inverse: x = (x_scaled - min_) / scale
function norm(x, sc) { return x * sc.scale + sc.min_; }
function denorm(xScaled, sc) { return (xScaled - sc.min_) / sc.scale; }

function $(...ids) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) return el;
  }
  return null;
}

function firstTensor(out) {
  if (!out) return null;
  if (Array.isArray(out)) return out[0];
  if (out.dataSync || out.data) return out;
  if (typeof out === "object") {
    const k = Object.keys(out)[0];
    return out[k];
  }
  return null;
}

async function main() {
  await tf.ready();

  // підтримка різних id
  const btnLoad = $("btnLoad", "load");
  const btnPredict = $("btnPredict", "pred");

  const lookbackEl = $("lookback"); // може бути null

  const lastValEl = $("lastVal", "last");
  const predValEl = $("predVal", "forecast");
  const diffValEl = $("diffVal", "diff");

  // якщо якихось елементів немає — покажемо конкретно що
  const missing = [];
  if (!btnLoad) missing.push("button id=btnLoad або load");
  if (!btnPredict) missing.push("button id=btnPredict або pred");
  if (!lastValEl) missing.push("div id=lastVal або last");
  if (!predValEl) missing.push("div id=predVal або forecast");
  if (!diffValEl) missing.push("div id=diffVal або diff");

  if (missing.length) {
    throw new Error("Не знайдені елементи в HTML: " + missing.join(", "));
  }

  // load demo data
  const demo = await loadJSON("./data/demo.json");
  const values = demo.values;
  const sc = demo.scaler;

  // lookback: або з input, або з scaler, або дефолт 5
  if (lookbackEl && sc.lookback != null) lookbackEl.value = sc.lookback;

  // last known
  lastValEl.textContent = values[values.length - 1].toFixed(4);

  // load model
  btnLoad.addEventListener("click", async () => {
    try {
      btnLoad.disabled = true;
      btnLoad.textContent = "Завантаження...";

      model = await tf.loadGraphModel("./model/model.json");

      btnLoad.textContent = "Модель завантажено";
      btnPredict.disabled = false;

      console.log("Model inputs:", model.inputs?.map(i => i.name));
      console.log("Model outputs:", model.outputs?.map(o => o.name));
    } catch (e) {
      console.error(e);
      alert("Помилка завантаження моделі: " + e.message);
      btnLoad.disabled = false;
      btnLoad.textContent = "Завантажити модель";
    }
  });

  // predict (ONLY executeAsync for LSTM graph)
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
      const windowScaled = windowRaw.map(v => norm(v, sc));
      const x = tf.tensor(windowScaled, [1, lookback, 1], "float32");

      const out = await model.executeAsync(x);
      const yTensor = firstTensor(out);
      if (!yTensor) throw new Error("Модель не повернула Tensor.");

      const yVal = (await yTensor.data())[0];

      const forecast = denorm(yVal, sc);
      const last = windowRaw[windowRaw.length - 1];
      const diff = forecast - last;

      predValEl.textContent = forecast.toFixed(4);
      diffValEl.textContent = (diff >= 0 ? "+" : "") + diff.toFixed(4);

      tf.dispose(x);
      if (Array.isArray(out)) out.forEach(t => t.dispose && t.dispose());
      else if (out?.dispose) out.dispose();
    } catch (e) {
      console.error(e);
      alert("Помилка: " + e.message);
    }
  });
}

main().catch(err => {
  console.error(err);
  alert("Помилка: " + err.message);
});
