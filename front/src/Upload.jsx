import { useDropzone } from "react-dropzone";
import upload from "./assets/bitmap.svg";
import { useState } from "react";

export default function DropZoneUpload({ setFiles }) {
  const [isUploading, setIsUploading] = useState(false);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "image/*": [".jpeg", ".jpg", ".png", ".gif", ".HEIC"],
    },
    onDrop: async (acceptedFiles) => {
      if (acceptedFiles.length === 0) return;
      setFiles((prev) => [...prev, ...acceptedFiles]);
    },
  });

  return (
    <div {...getRootProps()} style={dropzoneStyle}>
      <input {...getInputProps()} />
      <img src={upload} alt="Upload icon" style={{ width: "50px" }} />

      {isUploading ? (
        <p>Enviando imagens para o servidor...</p>
      ) : isDragActive ? (
        <p>Solte as imagens aqui...</p>
      ) : (
        <p>Arraste imagens aqui, ou clique para selecionar</p>
      )}
    </div>
  );
}

const dropzoneStyle = {
  border: "1px solid #ccc",
  borderRadius: "1rem",
  padding: "20px",
  textAlign: "center",
  cursor: "pointer",
};
