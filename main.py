import tempfile
import os
import io
import base64
import glob
import numpy as np
from flask import Flask, request, redirect, send_file, jsonify
from skimage import io as skio
from skimage.transform import resize
from model_loader import KaomojiNet, CATEGORIAS

app = Flask(__name__)

# Cargar modelo entrenado (kaomojinet.keras)
model = KaomojiNet("kaomojinet.keras")

main_html = """
<html>
<head>
    <meta charset="utf-8">
</head>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
<script>
  var mousePressed = false;
  var lastX, lastY;
  var ctx;
  var categorias = ["uwu", "owo", ">.<", "u.u", "T.T"];

  function InitThis() {
      ctx = document.getElementById('myCanvas').getContext("2d");

      var currentIndex = parseInt(localStorage.getItem('kaomoji_idx') || '0', 10);
      var actual = categorias[currentIndex % categorias.length];
      localStorage.setItem('kaomoji_idx', (currentIndex + 1) % categorias.length);

      document.getElementById('numero').value = actual;

      $('#myCanvas').mousedown(function (e) {
          mousePressed = true;
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
      });

      $('#myCanvas').mousemove(function (e) {
          if (mousePressed) {
              Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
          }
      });

      $('#myCanvas').mouseup(function (e) {
          mousePressed = false;
      });
      $('#myCanvas').mouseleave(function (e) {
          mousePressed = false;
      });
  }

  function Draw(x, y, isDown) {
      if (isDown) {
          ctx.beginPath();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 11;
          ctx.lineJoin = "round";
          ctx.moveTo(lastX, lastY);
          ctx.lineTo(x, y);
          ctx.closePath();
          ctx.stroke();
      }
      lastX = x; lastY = y;
  }

  function clearArea() {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      document.getElementById('resultado').innerText = '';
      renderProbabilidades({}, null);
  }

  function prepareImg() {
      var canvas = document.getElementById('myCanvas');
      document.getElementById('myImage').value = canvas.toDataURL('image/png');
  }

  function submitDrawing() {
      prepareImg();
      document.getElementById('uploadForm').submit();
  }

  function renderProbabilidades(probs, winner) {
      var html = '';
      for (var i = 0; i < categorias.length; i++) {
          var c = categorias[i];
          var val = probs[c] !== undefined ? probs[c] : 0.0;
          var label = (c === '>.<') ? '&gt;.&lt;' : c;
          var isWin = (winner && c === winner);
          var style = isWin ? 'font-weight: bold; color: green;' : 'color: #0055cc;';
          html += '<div style="' + style + '">' + label + ': ' + val.toFixed(2) + '%</div>';
      }
      document.getElementById('prob_list').innerHTML = html;
  }

  function predictDrawing() {
      var canvas = document.getElementById('myCanvas');
      var dataURL = canvas.toDataURL('image/png');
      document.getElementById('resultado').innerText = 'Analizando...';

      fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataURL })
      })
      .then(function(res) { return res.json(); })
      .then(function(data) {
          if (data.error) {
              document.getElementById('resultado').innerText = 'Error: ' + data.error;
              return;
          }
          if (data.clase === 'Lienzo vacío') {
              document.getElementById('resultado').innerText = 'Lienzo vacío';
              renderProbabilidades({}, null);
              return;
          }
          document.getElementById('resultado').innerText = 'Predicción: ' + data.clase + ' (' + data.confianza + '%)';
          renderProbabilidades(data.probabilidades || {}, data.clase);
      })
      .catch(function(err) {
          document.getElementById('resultado').innerText = 'Error al predecir';
          console.error(err);
      });
  }
</script>
<body onload="InitThis();">
    <div align="left">
        <img src="https://upload.wikimedia.org/wikipedia/commons/f/f7/Uni-logo_transparente_granate.png" width="300"/>
    </div>

    <div align="center">
        <h1>Adivinador de Kaomojis</h1>
        <p style="color: green; font-weight: bold; margin-top: 0;">Categorías: uwu, owo, &gt;.&lt;, u.u, T.T</p>
        <h2 id="resultado" style="min-height: 32px; margin: 10px 0;"></h2>

        <!-- Canvas exactamente centrado, con lista de probabilidades en azul a la derecha -->
        <div style="position: relative; width: 200px; margin: 0 auto;">
            <canvas id="myCanvas" width="200" height="200" style="border:2px solid black"></canvas>
            <div id="prob_panel" style="position: absolute; left: 220px; top: 10px; text-align: left; font-family: monospace; font-size: 15px; color: #0055cc; white-space: nowrap;">
                <b>Probabilidades</b>
                <div id="prob_list" style="margin-top: 8px; line-height: 1.6;">
                    <div style="color: #0055cc;">uwu: 0.0%</div>
                    <div style="color: #0055cc;">owo: 0.0%</div>
                    <div style="color: #0055cc;">&gt;.&lt;: 0.0%</div>
                    <div style="color: #0055cc;">u.u: 0.0%</div>
                    <div style="color: #0055cc;">T.T: 0.0%</div>
                </div>
            </div>
        </div>

        <br/>
        <form id="uploadForm" method="post" action="upload" onsubmit="prepareImg();" style="display: inline;">
            <input id="numero" name="numero" type="hidden" value="">
            <input id="myImage" name="myImage" type="hidden" value="">
            <button type="button" onclick="clearArea();">Borrar</button>
            <button type="button" onclick="submitDrawing();">Enviar</button>
            <button type="button" onclick="predictDrawing();">Predecir</button>
        </form>
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    return main_html

@app.route("/recolectar")
def recolectar():
    return main_html

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        img_data = data.get("image", "")
        if not img_data and "myImage" in request.form:
            img_data = request.form.get("myImage")

        if not img_data:
            return jsonify({"error": "No se recibio ninguna imagen"}), 400

        img_b64 = img_data.replace("data:image/png;base64,", "")
        img_bytes = base64.b64decode(img_b64)
        img_raw = skio.imread(io.BytesIO(img_bytes))

        if img_raw.ndim == 3 and img_raw.shape[2] == 4:
            alpha = img_raw[:, :, 3]
        elif img_raw.ndim == 3:
            alpha = 255 - np.mean(img_raw[:, :, :3], axis=2)
        else:
            alpha = img_raw

        if np.max(alpha) == 0:
            return jsonify({
                "clase": "Lienzo vacío",
                "confianza": 0.0,
                "probabilidades": {c: 0.0 for c in CATEGORIAS}
            })

        resized = resize(alpha, (64, 64), preserve_range=True) / 255.0

        resultado = model.predict(resized)
        return jsonify(resultado)

    except Exception as err:
        print("[predict] Error:", err)
        return jsonify({"error": str(err)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    try:
        img_data = request.form.get("myImage").replace("data:image/png;base64,", "")
        aleatorio = request.form.get("numero")
        with tempfile.NamedTemporaryFile(delete=False, mode="w+b", suffix=".png", dir=str(aleatorio)) as fh:
            fh.write(base64.b64decode(img_data))
        print(f"Imagen guardada en {aleatorio}")
    except Exception as err:
        print("Error en upload:", err)

    return redirect("/", code=302)

@app.route("/prepare", methods=["GET"])
def prepare_dataset():
    images = []
    digits = []
    for digit in CATEGORIAS:
        filelist = glob.glob(f"{digit}/*.png")
        if not filelist:
            continue
        images_read = skio.concatenate_images(skio.imread_collection(filelist))
        images_read = images_read[:, :, :, 3]
        digits_read = np.array([digit] * images_read.shape[0])
        images.append(images_read)
        digits.append(digits_read)
    if not images:
        return "No hay imagenes recolectadas aun", 400
    images = np.vstack(images)
    digits = np.concatenate(digits)
    np.save("X.npy", images)
    np.save("y.npy", digits)
    return "OK!"

@app.route("/X.npy", methods=["GET"])
def download_X():
    return send_file("./X.npy")

@app.route("/y.npy", methods=["GET"])
def download_y():
    return send_file("./y.npy")

for d in CATEGORIAS:
    os.makedirs(str(d), exist_ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
