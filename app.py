from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configurações
UPLOAD_FOLDER = "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif","heic"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

# Criar pasta de uploads se não existir
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nome de arquivo vazio"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Tipo de arquivo não permitido"}), 400

    try:
        # Ler a imagem
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Falha ao decodificar a imagem"}), 400

        # Processar a imagem
        from study import upscale

        args = {
            "fidelity_weight": 0.5,
            "upscale": 1,
            "has_aligned": False,
            "only_center_face": False,
            "draw_box": False,
            "detection_model": "retinaface_resnet50",
            "bg_upsampler": None,
            "face_upsample": False,
        }
        processed_img = upscale(img,**args)

        # Gerar nome único para o arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.name)
        filename = f"processed_{timestamp}_{original_filename}.png"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Salvar a imagem processada
        success = cv2.imwrite(save_path, processed_img)

        if not success:
            return jsonify({"error": "Falha ao salvar a imagem processada"}), 500

        return jsonify(
            {"success": True, "imageUrl": f"/uploads/{filename}", "filename": filename}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
