
let model = null;

async function loadJSON(p){ const r = await fetch(p); if(!r.ok) throw new Error(p); return await r.json(); }
function norm(x, sc){ return x*sc.scale + sc.min_; }
function denorm(x, sc){ return (x - sc.min_) / sc.scale; }

async function main(){
  const demo = await loadJSON("data/demo.json");
  const values = demo.values;
  const sc = demo.scaler;
  const lookback = sc.lookback;

  document.getElementById("last").textContent = values[values.length-1].toFixed(4);

  document.getElementById("load").onclick = async () => {
    const btn = document.getElementById("load");
    btn.disabled = true; btn.textContent="Завантаження...";
    model = await tf.loadGraphModel("model/model.json");
    btn.textContent="Модель завантажено";
    document.getElementById("pred").disabled = false;
  };

  document.getElementById("pred").onclick = async () => {
    const w = values.slice(values.length - lookback);
    const wScaled = w.map(v => norm(v, sc));
    const x = tf.tensor(wScaled, [1, lookback, 1], "float32");
    const y = model.predict(x);
    const yVal = (await y.data())[0];
    const forecast = denorm(yVal, sc);

    const last = w[w.length-1];
    const diff = forecast - last;

    document.getElementById("forecast").textContent = forecast.toFixed(4);
    document.getElementById("diff").textContent = (diff>=0?"+":"") + diff.toFixed(4);

    tf.dispose([x,y]);
  };
}

main().catch(e => { console.error(e); alert(e.message); });
