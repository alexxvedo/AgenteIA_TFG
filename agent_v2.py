import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from agent_functions import *

# Cargar variables de entorno
load_dotenv()

### ACTIVAR API ###
app = FastAPI()

# Configuración de API Key para Google Gemini desde variables de entorno
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️  ADVERTENCIA: GOOGLE_API_KEY no configurada")
    print("📝 Para configurar:")
    print("   1. Crea un archivo .env en python-agent/")
    print("   2. Añade: GOOGLE_API_KEY=tu-api-key-aqui")
    print("   3. Obtén tu API key en: https://makersuite.google.com/app/apikey")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google Gemini API configurada correctamente")

"""
def find_relevant_documents(question: str, collection_id: str, limit: int = 5) -> List[str]:
    \"""Encuentra documentos relevantes para una pregunta usando similitud de embeddings.\"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Error de conexión a la base de datos")
    
    # Obtener el embedding de la pregunta
    question_embedding = generate_embedding(question)
    
    # Buscar documentos similares
    cursor = conn.cursor()
    cursor.execute(\"\"\"
    SELECT content, cosine_similarity(embedding, %s) as similarity 
    FROM documents 
    WHERE collection_id = %s 
    ORDER BY similarity DESC 
    LIMIT %s
    \"\"\", (question_embedding, collection_id, limit))
    
    results = cursor.fetchall()
    conn.close()
    
    # Extraer solo el contenido de los documentos
    documents = [row[0] for row in results]
    return documents
"""

# Esquemas de datos
class RelevantLine(BaseModel):
    """Esquema para líneas relevantes de un documento."""
    line_number: int = Field(..., description="Número de línea en el documento")
    content: str = Field(..., description="Contenido de la línea")

class DocumentSource(BaseModel):
    """Esquema para documentos con información detallada."""
    document_id: str = Field(..., description="ID del documento")
    content: str = Field(..., description="Contenido completo del documento")
    file_name: str = Field(..., description="Nombre del archivo")
    file_type: str = Field(..., description="Tipo de archivo")
    similarity_score: float = Field(..., description="Puntuación de similitud con la pregunta")
    relevant_lines: List[RelevantLine] = Field(default_factory=list, description="Líneas relevantes del documento")

class Document(BaseModel):
    """Esquema para documentos."""
    content: Optional[str] = Field(None, description="Contenido del documento (texto plano)")
    pdf_base64: Optional[str] = Field(None, description="Contenido del PDF en base64")
    file_name: Optional[str] = Field(None, description="Nombre del archivo")
    desired_length: Optional[int] = Field(0, description="Longitud deseada del resumen en líneas")
    output_format: Optional[str] = Field("markdown", description="Formato de salida: 'markdown' o 'tiptap'")

class ConversationMessage(BaseModel):
    """Esquema para mensajes en una conversación."""
    role: str = Field(..., description="Rol del mensaje: 'user' o 'assistant'")
    content: str = Field(..., description="Contenido del mensaje")

class QuestionRequest(BaseModel):
    """Esquema para solicitudes de preguntas."""
    question: str = Field(..., description="Pregunta del usuario")
    collection_id: Optional[str] = Field(None, description="ID de la colección")
    similar_documents: Optional[List[Dict[str, Any]]] = Field(None, description="Documentos similares encontrados")
    additional_context: Optional[str] = Field(None, description="Contexto adicional proporcionado por el usuario")
    conversation_history: Optional[List[ConversationMessage]] = Field(default_factory=list, description="Historial de la conversación")

class QuestionAnswer(BaseModel):
    """Esquema para respuestas a preguntas."""
    answer: str = Field(..., description="Respuesta generada")
    sources: List[DocumentSource] = Field(default_factory=list, description="Fuentes utilizadas para generar la respuesta")
    confidence_score: float = Field(..., description="Puntuación de confianza en la respuesta")

class Collection(BaseModel):
    """Schema para colecciones."""
    id: str
    name: str
    description: Optional[str] = None
    owner_id: str


### ENDPOINTS ###


@app.post("/generate_embedding/", response_model=List[float])
def generate_embedding_endpoint(question: QuestionRequest):
    try:
        print(question.question)
        embedding = generate_embedding(question.question)
        print(f"Generated embedding with length: {len(embedding)}")
        return embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar embedding: {str(e)}")


@app.post("/process-document/", response_model=dict)
def process_document(document: Document):
    """
    Procesa un documento y lo almacena en la base de datos.
    
    Args:
        document: Documento a procesar (puede ser texto plano o PDF en base64)
        
    Returns:
        dict: Resultado del procesamiento
    """
    try:
        content = ""
        
        # Si se proporciona contenido de texto, usarlo directamente
        if document.content:
            content = document.content
        # Si se proporciona un PDF en base64, extraer el texto
        elif document.pdf_base64:
            try:
                content = extract_text_from_pdf_base64(document.pdf_base64)
                print(content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error al extraer texto del PDF: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Se debe proporcionar contenido de texto o un PDF en base64")
        
        # Generar el embedding para el contenido extraído
        embedding = generate_embedding(content)
        
        # Aquí se podría almacenar el documento y su embedding en una base de datos
        
        return {
            "status": "success", 
            "embedding": embedding,
            "content": content,
            "file_name": document.file_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el documento: {str(e)}")

@app.post("/generate-flashcards/", response_model=List[Flashcard])
def generate_flashcards(document: Document, num_flashcards: int = 5):
    """
    Genera flashcards a partir de un documento.
    
    Args:
        document: Documento del cual generar flashcards
        num_flashcards: Número de flashcards a generar (por defecto 5)
        
    Returns:
        List[Flashcard]: Lista de flashcards generadas
    """

    print(document)
    try:
        return generate_flashcards_content(document.content, num_flashcards)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar flashcards: {str(e)}")

@app.post("/generate-brief-summary/", response_model=BriefSummary)
def generate_brief_summary(document: Document):
    """
    Genera un resumen breve de un documento.
    
    Args:
        document: Documento a resumir
        
    Returns:
        BriefSummary: Resumen breve generado en formato Tiptap JSON
    """
    try:
        # Siempre usar formato Tiptap JSON, independientemente del formato solicitado
        output_format = "tiptap"
        
        print(f"Generando resumen breve con formato: {output_format}")
        return generate_brief_summary_content(document.content, output_format=output_format)
    except Exception as e:
        print(f"Error al generar resumen breve: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al generar resumen breve: {str(e)}")

@app.post("/generate-detailed-summary/", response_model=BriefSummary)
def generate_detailed_summary(document: Document, desired_length: int = 0):
    """
    Genera un resumen detallado de un documento con una longitud aproximada especificada.
    
    Args:
        document: Documento a resumir en detalle
        desired_length: Número aproximado de palabras deseadas para el resumen (opcional)
        
    Returns:
        BriefSummary: Resumen detallado generado en formato Tiptap JSON
    """
    try:
        # Obtener la longitud deseada del resumen, ya sea del parámetro de la URL o del cuerpo de la solicitud
        word_count = desired_length
        if document.desired_length and document.desired_length > 0:
            word_count = document.desired_length
        
        # Siempre usar formato Tiptap JSON, independientemente del formato solicitado
        output_format = "tiptap"
        
        print(f"Generando resumen detallado con longitud deseada: {word_count} palabras y formato: {output_format}")
        return generate_detailed_summary_content(document.content, word_count, output_format=output_format)
    except Exception as e:
        print(f"Error al generar resumen detallado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al generar resumen detallado: {str(e)}")

@app.post("/answer-question/", response_model=QuestionAnswer)
def answer_question(request: QuestionRequest):
    """
    Responde una pregunta usando documentos de contexto.
    
    Args:
        request: Solicitud con la pregunta, el ID de la colección y documentos similares
        
    Returns:
        QuestionAnswer: Respuesta generada con metadatos
    """
    try:
        return answer_question_with_context(
            question=request.question,
            collection_id=request.collection_id,
            similar_documents=request.similar_documents,
            additional_context=request.additional_context,
            conversation_history=request.conversation_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al responder la pregunta: {str(e)}")

@app.post("/answer-general-question/", response_model=QuestionAnswer)
def answer_general_question(request: QuestionRequest):
    """
    Responde una pregunta general sin contexto específico.
    
    Args:
        request: Solicitud con la pregunta
        
    Returns:
        QuestionAnswer: Respuesta generada
    """
    try:
        return answer_general_question_content(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al responder la pregunta general: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
