// web/js/app.js
let model = null;

async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Fetch failed: ${path} (${res.status})`);
  return await res.json();
}

// MinMaxScaler params saved from Python:
// x_scaled = x * scale + min_
// inverse: x = (x_scaled - min_) / scale
function norm(x, sc) {
  return x * sc.scale + sc.min_;
}
function denorm(xScaled, sc) {
  return (xScaled - sc.min_) / sc.scale;
}

function firstTensor(out) {
  if (!out) return null;
  if (Array.isArray(out)) return out[0];
  // Tensor has dataSync/data
  if (typeof out === "object" && (out.dataSync || out.data)) return out;
  // Some TFJS APIs may return map object
  if (typeof out === "object") {
    const k = Object.keys(out)[0];
    return out[k];
  }
  return null;
}

async function main() {
  await tf.ready();

  // Elements (must match your index.html ids)
  const btnLoad = document.getElementById("btnLoad");
  const btnPredict = document.getElementById("btnPredict");
  const lookbackEl = document.getElementById("lookback");

  const lastValEl = document.getElementById("lastVal");
  const predValEl = document.getElementById("predVal");
  const diffValEl = document.getElementById("diffVal");

  // Load demo data + scaler
  const demo = await loadJSON("./data/demo.json");
  const values = demo.values;
  const sc = demo.scaler;

  // If lookback stored in scaler.json – set input value
  if (sc.lookback != null) lookbackEl.value = sc.lookback;

  // Show last known value
  lastValEl.textContent = values[values.length - 1].toFixed(4);

  // Load model
  btnLoad.addEventListener("click", async () => {
    try {
      btnLoad.disabled = true;
      btnLoad.textContent = "Завантаження...";

      model = await tf.loadGraphModel("./model/model.json");

      btnLoad.textContent = "Модель завантажено";
      btnPredict.disabled = false;

      // Debug (optional)
      console.log("Model loaded");
      console.log("Inputs:", model.inputs?.map(i => i.name));
      console.log("Outputs:", model.outputs?.map(o => o.name));
    } catch (e) {
      console.error(e);
      alert("Помилка завантаження моделі: " + e.message);
      btnLoad.disabled = false;
      btnLoad.textContent = "Завантажити модель";
    }
  });

  // Predict (IMPORTANT: executeAsync for LSTM GraphModel)
  btnPredict.addEventListener("click", async () => {
    try {
      if (!model) {
        alert("Спочатку завантаж модель.");
        return;
      }

      const lookback = parseInt(lookbackEl.value, 10);
      if (!Number.isFinite(lookback) || lookback < 1) {
        alert("Невірний lookback.");
        return;
      }

      if (values.length < lookback) {
        alert(`Недостатньо даних: треба ${lookback}, є ${values.length}`);
        return;
      }

      const windowRaw = values.slice(values.length - lookback);
      const windowScaled = windowRaw.map(v => norm(v, sc));

      const x = tf.tensor(windowScaled, [1, lookback, 1], "float32");

      // ✅ executeAsync is required because model contains dynamic ops (while/Exit)
      const out = await model.executeAsync(x);
      const yTensor = firstTensor(out);
      if (!yTensor) throw new Error("Модель не повернула Tensor.");

      const yVal = (await yTensor.data())[0];

      const forecast = denorm(yVal, sc);
      const last = windowRaw[windowRaw.length - 1];
      const diff = forecast - last;

      predValEl.textContent = forecast.toFixed(4);
      diffValEl.textContent = (diff >= 0 ? "+" : "") + diff.toFixed(4);

      // Cleanup
      tf.dispose(x);
      if (Array.isArray(out)) out.forEach(t => t.dispose && t.dispose());
      else if (out && out.dispose) out.dispose();
    } catch (e) {
      console.error(e);
      alert("Помилка прогнозу: " + e.message);
    }
  });
}

main().catch(err => {
  console.error(err);
  alert("Init error: " + err.message);
});
