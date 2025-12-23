let model = null;

async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Fetch failed: ${path} (${res.status})`);
  return await res.json();
}

function norm(x, sc) { return x * sc.scale + sc.min_; }
function denorm(xScaled, sc) { return (xScaled - sc.min_) / sc.scale; }

async function main() {
  await tf.ready();

  const btnLoad = document.getElementById("btnLoad");
  const btnPredict = document.getElementById("btnPredict");
  const lookbackEl = document.getElementById("lookback");

  const lastValEl = document.getElementById("lastVal");
  const predValEl = document.getElementById("predVal");
  const diffValEl = document.getElementById("diffVal");

  const demo = await loadJSON("./data/demo.json");
  const values = demo.values;
  const sc = demo.scaler;

  if (sc.lookback != null) lookbackEl.value = sc.lookback;
  lastValEl.textContent = Number(values[values.length - 1]).toFixed(4);

  btnLoad.onclick = async () => {
    try {
      btnLoad.disabled = true;
      btnLoad.textContent = "Завантаження...";
      model = await tf.loadLayersModel("./model/model.json");   // ✅ layers
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

      const lookback = parseInt(lookbackEl.value, 10);
      const windowRaw = values.slice(values.length - lookback);
      const windowScaled = windowRaw.map(v => norm(Number(v), sc));

      const x = tf.tensor(windowScaled, [1, lookback, 1], "float32");

      const y = model.predict(x);               // ✅ predict
      const yVal = (await y.data())[0];

      const forecast = denorm(yVal, sc);
      const last = Number(windowRaw[windowRaw.length - 1]);
      const diff = forecast - last;

      predValEl.textContent = Number(forecast).toFixed(4);
      diffValEl.textContent = (diff >= 0 ? "+" : "") + Number(diff).toFixed(4);

      tf.dispose([x, y]);
    } catch (e) {
      console.error(e);
      alert("Помилка прогнозу: " + e.message);
    }
  };
}

window.addEventListener("DOMContentLoaded", () => {
  main().catch(e => {
    console.error(e);
    alert("Помилка: " + e.message);
  });
});

