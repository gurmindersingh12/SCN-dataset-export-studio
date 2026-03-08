from fastapi import FastAPI

app = FastAPI(title="SCN Dataset Export Studio")


@app.get("/")
def read_root():
    return {"message": "SCN Dataset Export Studio backend running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
