"""
FastAPI REST Server
===================

Production-ready API for AI image authenticity detection.
"""

from pathlib import Path
from typing import Optional
import io
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).parent.parent))


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    image_name: str
    prediction: str
    label: int
    confidence: float
    confidence_level: str
    probabilities: dict


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


def create_app(model_path: Optional[str] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Parameters
    ----------
    model_path : str, optional
        Path to trained model
        
    Returns
    -------
    FastAPI
        Configured application
    """
    app = FastAPI(
        title="AI Image Authenticity Checker API",
        description="Detect whether images are real photographs or AI-generated",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize predictor
    predictor = None
    
    @app.on_event("startup")
    async def startup():
        nonlocal predictor
        from inference.predict import ImagePredictor
        
        try:
            predictor = ImagePredictor(model_path=model_path)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            predictor = ImagePredictor()
    
    @app.get("/", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=predictor is not None and predictor.model is not None,
            version="1.0.0"
        )
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return await health_check()
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        explain: bool = Query(False, description="Include feature explanations")
    ):
        """
        Predict whether an uploaded image is real or AI-generated.
        
        Parameters
        ----------
        file : UploadFile
            Image file to analyze
        explain : bool
            Whether to include feature explanations
            
        Returns
        -------
        PredictionResponse
            Prediction result
        """
        if predictor is None or predictor.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate file type
        content_type = file.content_type
        if not content_type or not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        try:
            # Read file contents
            contents = await file.read()
            
            # Save to temp file
            suffix = Path(file.filename).suffix if file.filename else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            
            # Predict
            result = predictor.predict(tmp_path, return_features=explain)
            
            # Clean up
            Path(tmp_path).unlink()
            
            return PredictionResponse(
                image_name=file.filename or "uploaded_image",
                prediction=result.prediction,
                label=result.label,
                confidence=result.confidence,
                confidence_level=result.confidence_level,
                probabilities=result.probabilities
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch")
    async def predict_batch(files: list[UploadFile] = File(...)):
        """
        Predict multiple images at once.
        
        Parameters
        ----------
        files : List[UploadFile]
            Image files to analyze
            
        Returns
        -------
        List[PredictionResponse]
            Predictions for all images
        """
        if predictor is None or predictor.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        
        for file in files:
            try:
                contents = await file.read()
                suffix = Path(file.filename).suffix if file.filename else ".jpg"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(contents)
                    tmp_path = tmp.name
                
                result = predictor.predict(tmp_path)
                Path(tmp_path).unlink()
                
                results.append(PredictionResponse(
                    image_name=file.filename or "uploaded_image",
                    prediction=result.prediction,
                    label=result.label,
                    confidence=result.confidence,
                    confidence_level=result.confidence_level,
                    probabilities=result.probabilities
                ))
                
            except Exception as e:
                results.append({
                    "image_name": file.filename,
                    "error": str(e)
                })
        
        return results
    
    return app


# For running with uvicorn directly
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
