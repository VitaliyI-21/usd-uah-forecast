btnPredict.onclick = async () => {
  try {
    if (!model) return;

    const lookback = parseInt(document.getElementById("lookback").value, 10);

    const windowRaw = values.slice(values.length - lookback);
    const windowScaled = windowRaw.map(v => norm(Number(v), sc));

    const x = tf.tensor(windowScaled, [1, lookback, 1], "float32");

    // ✅ КЛЮЧОВИЙ ФІКС — передаємо feed dict
    const inputName = model.inputs[0].name; // наприклад "input_window:0"
    const feed = {};
    feed[inputName] = x;

    const out = await model.executeAsync(feed);

    const yTensor = Array.isArray(out) ? out[0] : out;
    const yVal = (await yTensor.data())[0];

    const forecast = denorm(yVal, sc);
    const last = Number(windowRaw[windowRaw.length - 1]);
    const diff = forecast - last;

    document.getElementById("predVal").textContent = forecast.toFixed(4);
    document.getElementById("diffVal").textContent =
      (diff >= 0 ? "+" : "") + diff.toFixed(4);

    tf.dispose(x);
    if (Array.isArray(out)) out.forEach(t => t.dispose());
    else out.dispose();

  } catch (e) {
    console.error(e);
    alert("Помилка прогнозу: " + e.message);
  }
};
