from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your Google API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(
    title="Invoice OCR API",
    description="API for processing invoice images using OCR",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Safety Settings of Model
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=MODEL_CONFIG,
    safety_settings=safety_settings
)

def image_format(image_path):
    """Format image for Gemini API"""
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")
    
    # Determine MIME type based on file extension
    extension = img.suffix.lower()
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp'
    }
    
    mime_type = mime_type_map.get(extension, 'image/jpeg')
    
    image_parts = [
        {
            "mime_type": mime_type,
            "data": img.read_bytes()
        }
    ]
    return image_parts

def gemini_output(image_path, system_prompt, user_prompt):
    """Generate output using Gemini model"""
    try:
        image_info = image_format(image_path)
        input_prompt = [system_prompt, image_info[0], user_prompt]
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_invoice(image_path):
    """Process invoice image and return parsed JSON."""
    system_prompt = """
    You are a specialist in comprehending import and export Invoices.
    Input images in the form of import and export invoices will be provided to you,
    and your task is to respond with a valid JSON object based on the content of the input image.
    """

    user_prompt = """Convert Invoice data into JSON format with this structure. Always return this exact JSON structure even if some values are missing:

    {
      "invoice_header": {
        "company_name": "",
        "address": "",
        "invoice_number": "",
        "date": "",
        "customer_details": {
          "name": "",
          "address": "",
          "gstin": ""
        }
      },
      "invoice_details": [
        {
          "item_number": "",
          "item_name": "",
          "quantity": "",
          "unit": "",
          "rate": "",
          "amount": ""
        }
      ],
      "totals": {
        "subtotal": "",
        "cgst_percentage": "",
        "cgst_amount": "",
        "sgst_percentage": "",
        "sgst_amount": "",
        "total": ""
      }
    }

    Do not wrap the response in triple backticks or Markdown. Respond with only JSON text.
    """

    output = gemini_output(image_path, system_prompt, user_prompt)
    
    # Clean triple backtick if accidentally included
    output = output.strip()
    if output.startswith("```json"):
        output = re.sub(r"^```json\n|```$", "", output)

    try:
        parsed = json.loads(output)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Gemini response. Error: {e}\nResponse:\n{output}")

@app.get("/")
def read_root():
    return {"message": "Welcome to Invoice OCR API"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (jpg, jpeg, png)"
            )

        # Save uploaded file
        file_path = UPLOADS_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the invoice
        result = process_invoice(str(file_path))

        # Clean up uploaded file
        os.remove(file_path)

        return JSONResponse(
            content={
                "status": "success",
                "data": result
            }
        )

    except Exception as e:
        # Clean up uploaded file in case of error
        if 'file_path' in locals():
            try:
                os.remove(file_path)
            except:
                pass
                
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)

