# Anonimization_of_medical_images
> Hackathon project — Visual de-identification of medical images.

---
## 🗂️ Project Structure

```
Anonimization_of_medical_images/
├── requirements.txt
├── README.md
└── Pipeline/
    └── OCR.py
```

---
## OCR de texto en imagenes

Script: `Pipeline/OCR.py`

Lee imagenes desde `Images/`, extrae texto con Tesseract OCR y guarda:

- Un archivo `.txt` por imagen con el texto detectado.
- Una imagen `*_boxed.png` con cajas alrededor de palabras detectadas.
- Una imagen `*_gray.png` en escala de grises usada en el preprocesamiento.

## Setup rapido (PowerShell)

1. Crear y activar entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Instalar dependencias Python:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Instalar Tesseract OCR (sistema operativo):

- Windows (winget):

```powershell
winget install UB-Mannheim.TesseractOCR
```

Si Python no encuentra Tesseract, ejecutar el script con:

```powershell
python Pipeline\OCR.py --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Ejecucion

```powershell
python Pipeline\OCR.py
```

Opcional:

```powershell
python Pipeline\OCR.py --input-dir Images --output-dir Pipeline\ocr_output --min-conf 60
```
