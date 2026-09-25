import os
import uvicorn

if __name__ == "__main__":
    print("Launching AURA FLUX.1 Studio...")
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)
