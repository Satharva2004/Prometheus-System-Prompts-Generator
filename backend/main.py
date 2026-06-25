from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from app.api.endpoints import ai, litcode, sap

app.include_router(ai.router, prefix="/api/ai", tags=["ai"])
app.include_router(litcode.router, prefix="/api/litcode", tags=["litcode"])
app.include_router(sap.router, prefix="/api/sap", tags=["sap"])

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
