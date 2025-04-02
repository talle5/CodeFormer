import React, { useState } from "react";
import axios from "axios";
import DropZoneUpload from "./Upload.jsx";
import close from "./assets/close.png";
import "./App.css";

function App() {
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  const handleUpload = async () => {
    if (files.length === 0) {
      setError("Por favor, selecione um arquivo");
      return;
    }

    setError(null);
    setIsUploading(true);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", files[0]);

      const response = await axios.post(
        "http://127.0.0.1:5000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percentCompleted);
          },
        }
      );

      const { imageUrl, filename } = response.data;

      // Atualiza o estado com a URL completa da imagem processada
      setFiles([`http://localhost:5000${imageUrl}`]);
    } catch (err) {
      const errorMsg =
        err.response?.data?.error ||
        err.message ||
        "Erro desconhecido no upload";
      setError(errorMsg);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <>
      <section style={{ display: "flex", gap: "2rem", padding: "2rem" }}>
        <div style={{ width: "100%" }}>
          <DropZoneUpload setFiles={setFiles} />

          {files.length > 0 && (
            <button
              onClick={handleUpload}
              disabled={isUploading}
              style={{ marginTop: "1rem" }}
            >
              {isUploading ? "Enviando..." : "Enviar para Processamento"}
            </button>
          )}

          {isUploading && (
            <div style={{ marginTop: "1rem" }}>
              <progress value={progress} max="100" />
              <p>{progress}% concluído</p>
            </div>
          )}

          {error && (
            <div style={{ color: "red", marginTop: "1rem" }}>{error}</div>
          )}
          <div></div>
        </div>
        <table style={{ borderBottomWidth: "0px" }}>
          <tbody></tbody>
        </table>
      </section>
      <section style={{ width: "100%" }}>
        <div style={{height:"100%"}}>
          <h2>Visualizador</h2>
          <Visualizador files={files} />
        </div>
      </section>
    </>
  );
}

const Visualizador = ({ files }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const handlePrevious = () => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : files.length - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev < files.length - 1 ? prev + 1 : 0));
  };

  if (files.length === 0) {
    return <p>Nenhuma imagem carregada ainda.</p>;
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "1rem",
        width: "100%",
      }}
    >
      <div
        style={{
          border: "1px solid #ccc",
          padding: "1rem",
          minHeight: "300px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <img
          src={files[currentIndex]}
          alt={`Imagem processada ${currentIndex + 1}`}
          style={{ maxWidth: "100%", maxHeight: "300px" }}
        />
      </div>

      <div style={{ display: "flex", justifyContent: "center", gap: "1rem" }}>
        <button onClick={handlePrevious} disabled={files.length <= 1}>
          Anterior
        </button>
        <span>
          {currentIndex + 1} / {files.length}
        </span>
        <button onClick={handleNext} disabled={files.length <= 1}>
          Próxima
        </button>
      </div>
    </div>
  );
};

const SelectedImages = ({ imagems }) => {
  return (
    <table style={{ borderBottomWidth: "0px" }}>
      <tbody>
        {imagems.map((img) => (
          <tr onClick={() => handleRowClick(img.id)}>
            <td>
              <img src={img} />
            </td>
            <td>
              <img src={close} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default App;
