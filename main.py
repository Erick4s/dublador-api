from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torchaudio
import os
import torch
from speechbrain.pretrained import SpeakerRecognition

app = FastAPI()

# Carrega modelo de verificação de locutor
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec")

# Pasta com áudios dos dubladores cadastrados
REFERENCE_FOLDER = "voices"

# Carrega embeddings dos dubladores uma vez
reference_embeddings = {}

def compute_embedding(file_path):
    signal, fs = torchaudio.load(file_path)
    embedding = verification.encode_batch(signal)
    return embedding.squeeze(0)

# Pré-carrega os embeddings
for filename in os.listdir(REFERENCE_FOLDER):
    if filename.endswith(".wav"):
        name = filename.replace(".wav", "")
        path = os.path.join(REFERENCE_FOLDER, filename)
        reference_embeddings[name] = compute_embedding(path)

@app.post("/identificar")
async def identificar(file: UploadFile = File(...)):
    try:
        with open("temp.wav", "wb") as f:
            f.write(await file.read())

        signal, fs = torchaudio.load("temp.wav")
        audio_embedding = verification.encode_batch(signal).squeeze(0)

        # Calcula similaridade com cada dublador
        resultados = []
        for nome, emb_ref in reference_embeddings.items():
            score = torch.nn.functional.cosine_similarity(audio_embedding, emb_ref, dim=0).item()
            resultados.append((nome, score))

        resultados.sort(key=lambda x: x[1], reverse=True)
        melhor = resultados[0]

        return {"dublador": melhor[0], "confiança": round(melhor[1], 4)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})
