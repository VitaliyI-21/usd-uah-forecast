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
  return out;
}

async function main() {
  await tf.ready();

  const btnLoad = document.getElementById("btnLoad");
  const btnPredict = document.getElementById("btnPredict");

  const lastValEl = document.getElementById("lastVal");
  const predValEl = document.getElementById("predVal");
  const diffValEl = document.getElementById("diffVal");

  if (!btnLoad || !btnPredict || !lastValEl || !predValEl || !diffValEl) {
    throw new Error("Перевір index.html: потрібні id btnLoad, btnPredict, lastVal, predVal, diffVal.");
  }

  const demo = await loadJSON("./data/demo.json");
  const values = demo.values;
  const sc = demo.scaler;
  const lookback = parseInt(sc.lookback ?? 5, 10);

  lastValEl.textContent = Number(values[values.length - 1]).toFixed(4);

  btnLoad.onclick = async () => {
    try {
      btnLoad.disabled = true;
      btnLoad.textContent = "Завантаження...";
      model = await tf.loadGraphModel("./model/model.json");
      btnLoad.textContent = "Модель завантажено";
      btnPredict.disabled = false;
    } catch (e) {
      console.error(e);
      alert("Помилка завантаження моделі: " + e.message);
      btnLoad.disabled = false;
      btnLoad.textContent = "Завантажити модель";
    }
  };

  btnPredict.onclick = async () => {
    try {
      if (!model) return;

      const windowRaw = values.slice(values.length - lookback);
      const windowScaled = windowRaw.map(v => norm(Number(v), sc));

      const x = tf.tensor(windowScaled, [1, lookback, 1], "float32");

      // ✅ LSTM graph => тільки executeAsync (інакше dynamic op Exit)
      const out = await model.executeAsync(x);
      const yTensor = firstTensor(out);
      const yVal = (await yTensor.data())[0];

      const forecast = denorm(yVal, sc);
      const last = Number(windowRaw[windowRaw.length - 1]);
      const diff = forecast - last;

      predValEl.textContent = Number(forecast).toFixed(4);
      diffValEl.textContent = (diff >= 0 ? "+" : "") + Number(diff).toFixed(4);

      tf.dispose(x);
      if (Array.isArray(out)) out.forEach(t => t.dispose && t.dispose());
      else if (out?.dispose) out.dispose();
    } catch (e) {
      console.error(e);
      alert("Помилка прогнозу: " + e.message);
    }
  };
}

window.addEventListener("DOMContentLoaded", () => {
  main().catch(err => {
    console.error(err);
    alert("Помилка: " + err.message);
  });
});
