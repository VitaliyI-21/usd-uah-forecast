let model = null;

async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Fetch failed: ${path} (${res.status})`);
  return await res.json();
}

function norm(x, sc) { return x * sc.scale + sc.min_; }
function denorm(x, sc) { return (x - sc.min_) / sc.scale; }

async function main() {
  await tf.ready();

  const btnLoad = document.getElementById("btnLoad");
  const btnPred = document.getElementById("btnPredict");

  const lastEl = document.getElementById("lastVal");
  const predEl = document.getElementById("predVal");
  const diffEl = document.getElementById("diffVal");
  const lookbackEl = document.getElementById("lookback");

  const demo = await loadJSON("data/demo.json");
  const values = demo.values;
  const sc = demo.scaler;

  lastEl.textContent = values[values.length - 1].toFixed(4);

  btnLoad.onclick = async () => {
    btnLoad.disabled = true;
    btnLoad.textContent = "Завантаження...";
    model = await tf.loadGraphModel("model/model.json");
    btnLoad.textContent = "Модель завантажено";
    btnPred.disabled = false;
  };

  btnPred.onclick = async () => {
    if (!model) return;

    const lookback = parseInt(lookbackEl.value, 10);
    if (values.length < lookback) {
      alert(`Недостатньо даних: треба ${lookback}, є ${values.length}`);
      return;
    }

    const windowRaw = values.slice(values.length - lookback);
    const windowScaled = windowRaw.map(v => norm(v, sc));

    const x = tf.tensor(windowScaled, [1, lookback, 1], "float32");

    // ✅ тільки executeAsync для LSTM GraphModel
    const out = await model.executeAsync(x);
    const yTensor = Array.isArray(out) ? out[0] : out;
    const yVal = (await yTensor.data())[0];

    const forecast = denorm(yVal, sc);
    const last = windowRaw[windowRaw.length - 1];
    const diff = forecast - last;

    predEl.textContent = forecast.toFixed(4);
    diffEl.textContent = (diff >= 0 ? "+" : "") + diff.toFixed(4);

    tf.dispose(x);
    if (Array.isArray(out)) out.forEach(t => t.dispose());
    else out.dispose();
  };
}

main().catch(err => {
  console.error(err);
  alert("Помилка: " + err.message);
});
