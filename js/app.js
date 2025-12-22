let model = null;

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Fetch failed: ${path} (${res.status})`);
  return await res.json();
}

function norm(x, sc) { return x * sc.scale + sc.min_; }
function denorm(x, sc) { return (x - sc.min_) / sc.scale; }

function firstTensor(output) {
  // executeAsync може повернути Tensor, масив Tensor або об'єкт
  if (Array.isArray(output)) return output[0];
  if (output && typeof output === "object" && output.dataSync == null) {
    const k = Object.keys(output)[0];
    return output[k];
  }
  return output;
}

async function main() {
  await tf.ready();

  const btnLoad = document.getElementById("load");
  const btnPred = document.getElementById("pred");

  const lastEl = document.getElementById("last");
  const predEl = document.getElementById("forecast");
  const diffEl = document.getElementById("diff");

  const demo = await loadJSON("./data/demo.json");
  const values = demo.values;
  const sc = demo.scaler;
  const lookback = parseInt(sc.lookback, 10);

  lastEl.textContent = values[values.length - 1].toFixed(4);

  btnLoad.onclick = async () => {
    btnLoad.disabled = true;
    btnLoad.textContent = "Завантаження...";
    model = await tf.loadGraphModel("./model/model.json");
    btnLoad.textContent = "Модель завантажено";
    btnPred.disabled = false;

    // Діагностика (можеш потім прибрати)
    console.log("Model inputs:", model.inputs?.map(x => x.name));
    console.log("Model outputs:", model.outputs?.map(x => x.name));
  };

  btnPred.onclick = async () => {
    if (!model) return;

    if (values.length < lookback) {
      alert(`Недостатньо даних: треба ${lookback}, є ${values.length}`);
      return;
    }

    const windowRaw = values.slice(values.length - lookback);
    const windowScaled = windowRaw.map(v => norm(v, sc));

    const x = tf.tensor(windowScaled, [1, lookback, 1], "float32");

    // Найстабільніший спосіб для GraphModel — executeAsync з іменем входу
    const inputName = (model.inputs && model.inputs[0] && model.inputs[0].name) ? model.inputs[0].name : null;

    let out;
    if (inputName) {
      const feed = {};
      feed[inputName] = x;
      out = await model.executeAsync(feed);
    } else {
      // fallback
      out = model.predict(x);
    }

    const yTensor = firstTensor(out);
    const yVal = (await yTensor.data())[0];

    const forecast = denorm(yVal, sc);
    const last = windowRaw[windowRaw.length - 1];
    const diff = forecast - last;

    predEl.textContent = forecast.toFixed(4);
    diffEl.textContent = (diff >= 0 ? "+" : "") + diff.toFixed(4);

    tf.dispose([x]);
    if (Array.isArray(out)) out.forEach(t => t.dispose());
    else if (out && out.dispose) out.dispose();
  };
}

main().catch(err => {
  console.error(err);
  alert("Помилка: " + err.message);
});
