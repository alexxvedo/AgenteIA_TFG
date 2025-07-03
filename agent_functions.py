import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional
import json
import google.generativeai as genai
from google.generativeai import types
from pydantic import BaseModel, Field
import PyPDF2
import io
import base64

# Definición de modelos de datos
# En lugar de importarlos de agent_v2, los definimos aquí para evitar importaciones circulares
class Flashcard(BaseModel):
    """Schema para flashcards."""
    question: str
    answer: str
    difficulty: int
    topic: str

class BriefSummary(BaseModel):
    """Schema para resúmenes breves."""
    summary: str

class DetailedSummary(BaseModel):
    """Schema para resúmenes detallados."""
    main_points: List[str]
    details: List[str]
    conclusions: List[str]

class QuestionAnswer(BaseModel):
    """Schema para respuestas a preguntas."""
    answer: str
    confidence: float
    sources: Optional[List[str]] = None

"""
Este módulo contiene las funciones principales para el agente de IA que utiliza
modelos de embedding y generación de texto para realizar diversas tareas como:
- Generación de embeddings para textos
- Creación de flashcards de estudio
- Generación de resúmenes (breves y detallados)
- Respuesta a preguntas con y sin contexto específico
"""

# Configuración del modelo de embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)



### FUNCTION DECLARATIONS ###

def create_function_schema(model_class):
    """
    Crea un schema compatible con Gemini a partir de una clase Pydantic.
    Elimina los campos que no son soportados por Gemini.
    """
    schema = model_class.model_json_schema()
    # Eliminar campos no soportados por Gemini
    schema.pop("title", None)
    schema.pop("description", None)
    return schema

create_flashcards = types.FunctionDeclaration(
    name="create_flashcards",
    description="Crea flashcards a partir de un texto para ayudar en el estudio",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Texto del cual se generarán las flashcards"
            },
            "num_flashcards": {
                "type": "integer",
                "description": "Número de flashcards a generar"
            }
        },
        "required": ["text"]
    }
)

create_brief_summary = types.FunctionDeclaration(
    name="create_brief_summary",
    description="Crea un resumen breve y conciso de un texto, usa el formato {output_format} y lo que necesites para hacerlo mejor",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Texto del cual se generará el resumen"
            }
        },
        "required": ["text"]
    }
)

create_detailed_summary = types.FunctionDeclaration(
    name="create_detailed_summary",
    description="Crea un resumen detallado con puntos principales, usa el formato {output_format} y lo que necesites para hacerlo mejor. Hazlo extremadamente detallado, orientado al estudio, sin dejar detalles importantes sin tocar. ",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Texto del cual se generará el resumen detallado"
            }
        },
        "required": ["text"]
    }
)

answer_question = types.FunctionDeclaration(
    name="answer_question",
    description="Responde una pregunta usando documentos de contexto",
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Pregunta a responder"
            },
            "collection_id": {
                "type": "string",
                "description": "ID de la colección de documentos a consultar"
            }
        },
        "required": ["question", "collection_id"]
    }
)

answer_general_question = types.FunctionDeclaration(
    name="answer_general_question",
    description="Responde una pregunta general sin contexto específico",
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Pregunta general a responder"
            }
        },
        "required": ["question"]
    }
)


def generate_embedding(text: str) -> List[float]:
    """
    Genera un vector de embedding para un texto utilizando un modelo de transformers.
    
    Este embedding es una representación numérica del texto que captura su significado
    semántico y puede ser utilizado para comparar la similitud entre diferentes textos.
    
    Args:
        text (str): El texto para el cual se generará el embedding.
        
    Returns:
        List[float]: Un vector de números flotantes que representa el embedding del texto.
    """
    # Tokeniza el texto y prepara los inputs para el modelo
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Desactiva el cálculo de gradientes para ahorrar memoria y acelerar la inferencia
    with torch.no_grad():
        # Obtiene las salidas del modelo
        outputs = model(**inputs)
    
    # Calcula el embedding promediando las representaciones de la última capa oculta
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    # Convierte el tensor a una lista de Python
    return embeddings[0].numpy().tolist()


### HERRAMIENTAS PARA EL MODELO GEMINI ###

# Definición de las herramientas disponibles para el modelo Gemini
# Estas herramientas permiten al modelo realizar acciones específicas
agent_tools = types.Tool(
    function_declarations=[
        create_flashcards,        # Crear tarjetas de estudio
        create_brief_summary,     # Generar resumen breve
        create_detailed_summary,  # Generar resumen detallado
        answer_question,          # Responder pregunta con contexto
        answer_general_question   # Responder pregunta general
    ]
)

def create_gemini_model(max_output_tokens=None):
    """
    Inicializa y devuelve un modelo Gemini para la generación de texto.
    
    Utiliza la versión 'gemini-2.5-flash' que ofrece un buen equilibrio entre
    velocidad y calidad para tareas de generación de texto.
    
    Args:
        max_output_tokens (int, optional): Número máximo de tokens de salida.
            Si es None, se usará el valor predeterminado de 32768 (máximo).
    
    Returns:
        GenerativeModel: Una instancia del modelo Gemini lista para usar.
    """
    # Configurar el modelo con el máximo número de tokens de salida
    # Gemini 1.5 Flash soporta hasta 32k tokens de salida
    if max_output_tokens is None:
        max_output_tokens = 32768  # Usar el máximo posible
    
    generation_config = genai.GenerationConfig(
        max_output_tokens=max_output_tokens,
        temperature=0.2,  # Temperatura baja para respuestas más deterministas
        top_p=0.95,
        top_k=40
    )
    
    model = genai.GenerativeModel(
        'gemini-2.0-flash',
        generation_config=generation_config
    )
    
    return model

def process_gemini_response(model, prompt: str):
    """
    Procesa una interacción con Gemini usando el sistema de chat y function calling.
    
    Esta función envía un prompt al modelo Gemini y procesa su respuesta, manejando
    tanto respuestas de texto como llamadas a funciones.
    
    Args:
        model: Modelo de Gemini inicializado.
        prompt (str): Prompt inicial del usuario o instrucción para el modelo.
        
    Returns:
        dict o str: Si el modelo decide llamar a una función, devuelve los argumentos
                   como un diccionario. De lo contrario, devuelve el texto generado.
    """
    try:
        print("Enviando prompt a Gemini:", prompt[:100] + "..." if len(prompt) > 100 else prompt)
        
        # Genera contenido con el modelo, configurando las herramientas disponibles
        result = model.generate_content(
            contents=prompt,
            generation_config=types.GenerationConfig(
                temperature=0.2  # Temperatura baja para respuestas más deterministas pero con algo de creatividad
            )
        )
        
        print("Respuesta recibida de Gemini")
        
        # Verificar si hay candidatos en la respuesta
        if not result.candidates or not result.candidates[0].content or not result.candidates[0].content.parts:
            print("No se recibieron candidatos válidos en la respuesta")
            return []  # Devolver una lista vacía si no hay candidatos
            
        # Obtener la primera parte de la respuesta
        response_part = result.candidates[0].content.parts[0]
        
        # Verificar si es una llamada a función
        if hasattr(response_part, 'function_call') and response_part.function_call:
            function_call = response_part.function_call
            print(f"Function called: {function_call.name}")
            print(json.dumps(function_call.args, indent=2))
            return function_call.args  # Devuelve los argumentos de la función
        
        # Si es texto, procesarlo como JSON si es posible
        text_response = response_part.text
        print("Texto de respuesta:", text_response[:100] + "..." if len(text_response) > 100 else text_response)
        
        # Limpiar el texto si está en un bloque de código markdown
        if text_response.strip().startswith("```") and "```" in text_response:
            # Extraer el contenido dentro de los bloques de código
            code_blocks = text_response.split("```")
            if len(code_blocks) >= 3:  # Al menos debe haber un bloque completo (inicio, contenido, fin)
                # El contenido está en el índice 1 (después del primer ```)
                cleaned_text = code_blocks[1]
                # Eliminar el identificador de lenguaje si existe (ej: "json")
                if cleaned_text.strip() and cleaned_text.strip()[0].isalpha():
                    cleaned_text = cleaned_text.strip().split("\n", 1)[1] if "\n" in cleaned_text.strip() else ""
                text_response = cleaned_text
                print("Texto limpiado de marcadores markdown:", text_response[:100] + "..." if len(text_response) > 100 else text_response)
        
        # Intentar parsear como JSON si parece ser un formato de lista de flashcards
        if text_response.strip().startswith('[') and text_response.strip().endswith(']'):
            try:
                parsed_json = json.loads(text_response)
                print(f"JSON parseado correctamente, se encontraron {len(parsed_json)} elementos")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"Error al decodificar JSON: {str(e)}")
                print("No se pudo decodificar la respuesta como JSON, tratando como texto")
        
        # Formato manual para flashcards (fallback)
        if "Flashcard" in text_response or "pregunta" in text_response.lower():
            # Implementar un parser simple para extraer flashcards del texto
            return parse_flashcards_from_text(text_response)
            
        # Si no es JSON ni tiene formato de flashcards, devolver el texto tal cual
        return text_response
        
    except Exception as e:
        print(f"Error al procesar la respuesta de Gemini: {str(e)}")
        import traceback
        traceback.print_exc()
        # Devolver una lista vacía en caso de error
        return []

def parse_flashcards_from_text(text):
    """
    Parsea flashcards de un texto no estructurado.
    
    Args:
        text (str): Texto que contiene información de flashcards
        
    Returns:
        list: Lista de diccionarios con información de flashcards
    """
    # Lista para almacenar las flashcards extraídas
    flashcards = []
    
    # Dividir por líneas y buscar patrones de flashcards
    lines = text.split('\n')
    current_card = {}
    
    for line in lines:
        line = line.strip()
        
        # Detectar inicio de nueva flashcard (número o "Flashcard")
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')) or "Flashcard" in line:
            # Guardar la flashcard anterior si existe
            if current_card and 'question' in current_card and 'answer' in current_card:
                # Asegurarse de que tenga los campos requeridos
                if 'difficulty' not in current_card:
                    current_card['difficulty'] = 1
                if 'topic' not in current_card:
                    current_card['topic'] = "General"
                flashcards.append(current_card)
            
            # Iniciar nueva flashcard
            current_card = {}
            continue
            
        # Detectar pregunta
        if line.startswith(('Pregunta:', 'Pregunta :', 'Q:', 'Question:')) or 'pregunta:' in line.lower():
            question_text = line.split(':', 1)[1].strip() if ':' in line else line
            current_card['question'] = question_text
            continue
            
        # Detectar respuesta
        if line.startswith(('Respuesta:', 'Respuesta :', 'A:', 'Answer:')) or 'respuesta:' in line.lower():
            answer_text = line.split(':', 1)[1].strip() if ':' in line else line
            current_card['answer'] = answer_text
            continue
            
        # Detectar dificultad
        if 'dificultad:' in line.lower() or 'difficulty:' in line.lower():
            try:
                difficulty_text = line.split(':', 1)[1].strip()
                # Extraer el número de dificultad
                import re
                difficulty_match = re.search(r'\d+', difficulty_text)
                if difficulty_match:
                    current_card['difficulty'] = int(difficulty_match.group())
                else:
                    current_card['difficulty'] = 1
            except:
                current_card['difficulty'] = 1
            continue
            
        # Detectar tema
        if 'tema:' in line.lower() or 'topic:' in line.lower():
            topic_text = line.split(':', 1)[1].strip() if ':' in line else line
            current_card['topic'] = topic_text
            continue
    
    # Añadir la última flashcard si existe
    if current_card and 'question' in current_card and 'answer' in current_card:
        if 'difficulty' not in current_card:
            current_card['difficulty'] = 1
        if 'topic' not in current_card:
            current_card['topic'] = "General"
        flashcards.append(current_card)
    
    return flashcards

def handle_function_call(function_name: str, params: dict):
    """
    Maneja la ejecución de funciones llamadas por el modelo Gemini.
    
    Esta función actúa como un dispatcher que ejecuta la función apropiada
    basada en el nombre de la función proporcionado por el modelo.
    
    Args:
        function_name (str): Nombre de la función a ejecutar.
        params (dict): Parámetros para la función.
        
    Returns:
        dict: Resultado de la función ejecutada en formato de diccionario.
        
    Raises:
        ValueError: Si la función especificada no existe.
    """
    # Maneja la creación de flashcards
    if function_name == "create_flashcards":
        # Crea flashcards de ejemplo (en una implementación real, generaría flashcards basadas en el texto)
        flashcards = []
        for i in range(params.get("num_flashcards", 5)):
            flashcards.append(Flashcard(
                question=f"Pregunta {i+1}",
                answer=f"Respuesta {i+1}",
                difficulty=1,
                topic="General"
            ))
        return [flashcard.model_dump() for flashcard in flashcards]  # Cambiado de dict() a model_dump()

    # Maneja la creación de resumen breve
    elif function_name == "create_brief_summary":
        return BriefSummary(
            summary="Resumen breve del texto proporcionado."
        ).model_dump()  # Cambiado de dict() a model_dump()

    # Maneja la creación de resumen detallado
    elif function_name == "create_detailed_summary":
        return BriefSummary(
            summary="Resumen detallado del texto proporcionado."
        ).model_dump()  # Cambiado de dict() a model_dump()

    # Maneja la respuesta a preguntas con contexto específico
    elif function_name == "answer_question":
        # Buscar documentos relevantes usando embeddings
        try:
            # Encuentra documentos relevantes para la pregunta
            # Esta función debe ser implementada para buscar en la base de datos
            # Por ahora, simulamos que no encontramos documentos
            relevant_docs = []  # find_relevant_documents() debería estar implementada

            # Si no se encontraron documentos relevantes
            if not relevant_docs:
                return QuestionAnswer(
                    answer="No encontré información relevante en la colección.",
                    confidence=0.0,
                    sources=[]
                ).model_dump()  # Cambiado de dict() a model_dump()

            # Crear un prompt con el contexto encontrado
            context = "\n".join(relevant_docs)
            model = create_gemini_model()
            prompt = f"""
            Usando este contexto:
            {context}

            Responde esta pregunta:
            {params['question']}
            """

            # Procesa la respuesta del modelo
            result = process_gemini_response(model, prompt)
            
            # Retorna la respuesta con metadatos
            return QuestionAnswer(
                answer=result,
                confidence=0.8,  # Confianza estimada
                sources=relevant_docs  # Documentos usados como fuente
            ).model_dump()  # Cambiado de dict() a model_dump()

        except Exception as e:
            # Manejo de errores
            return QuestionAnswer(
                answer=f"Error al buscar documentos: {str(e)}",
                confidence=0.0,
                sources=[]
            ).model_dump()  # Cambiado de dict() a model_dump()

    # Maneja la respuesta a preguntas generales sin contexto específico
    elif function_name == "answer_general_question":
        return QuestionAnswer(
            answer="Respuesta a la pregunta general.",
            confidence=0.9,
            sources=None
        ).model_dump()  # Cambiado de dict() a model_dump()

    # Si la función no existe
    else:
        raise ValueError(f"Función {function_name} no encontrada")


### FUNCIONES DE PROCESAMIENTO PARA TAREAS ESPECÍFICAS ###

def generate_flashcards_content(text: str, num_flashcards: int = 5) -> List[Flashcard]:
    """
    Genera flashcards de estudio a partir de un texto utilizando el modelo Gemini.
    
    Esta función crea tarjetas de estudio con preguntas y respuestas basadas
    en el contenido del texto proporcionado.
    
    Args:
        text (str): El texto del cual se generarán las flashcards.
        num_flashcards (int, optional): Número de flashcards a generar. Default: 5.
        
    Returns:
        List[Flashcard]: Lista de objetos Flashcard generados.
    """
    try:
        print(f"Generando {num_flashcards} flashcards a partir del texto")
        
        # Inicializa el modelo Gemini
        model = create_gemini_model()
        
        # Crea un prompt instructivo para el modelo
        prompt = f"""
        Crea {num_flashcards} flashcards de estudio del siguiente texto.
        Cada flashcard debe tener una pregunta clara, una respuesta concisa,
        un nivel de dificultad (1-3) y un tema.
        
        Devuelve las flashcards en formato JSON como una lista de objetos con los campos:
        question, answer, difficulty (número del 1 al 3), y topic.
        
        Ejemplo de formato:
        [
            {{
                "question": "¿Cuál es la capital de Francia?",
                "answer": "París",
                "difficulty": 1,
                "topic": "Geografía"
            }},
            {{
                "question": "¿Quién escribió Don Quijote?",
                "answer": "Miguel de Cervantes",
                "difficulty": 2,
                "topic": "Literatura"
            }}
        ]

        Texto: {text}
        """

        # Obtiene la respuesta del modelo
        result = process_gemini_response(model, prompt)
        
        print(f"Resultado obtenido: {type(result)}")
        
        # Si el resultado es una lista, convertirla a objetos Flashcard
        if isinstance(result, list):
            # Verificar que cada elemento tenga los campos necesarios
            valid_flashcards = []
            for card_data in result:
                if isinstance(card_data, dict) and 'question' in card_data and 'answer' in card_data:
                    # Asegurarse de que tenga los campos requeridos
                    if 'difficulty' not in card_data:
                        card_data['difficulty'] = 1
                    if 'topic' not in card_data:
                        card_data['topic'] = "General"
                    
                    # Crear objeto Flashcard
                    try:
                        flashcard = Flashcard(**card_data)
                        valid_flashcards.append(flashcard)
                    except Exception as e:
                        print(f"Error al crear flashcard: {str(e)}")
            
            return valid_flashcards
        else:
            # Si no es una lista, devolver una lista vacía
            print("El resultado no es una lista, devolviendo lista vacía")
            return []
            
    except Exception as e:
        print(f"Error al generar flashcards: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def markdown_to_tiptap_json(markdown_text: str) -> str:
    """
    Convierte texto markdown a formato JSON de Tiptap con estructura adecuada
    utilizando markdown-it-py para un análisis más robusto.
    
    Args:
        markdown_text (str): Texto en formato markdown
        
    Returns:
        str: JSON en formato Tiptap estructurado
    """
    try:
        from markdown_it import MarkdownIt
        import re
        
        # Si no hay texto, devolver documento vacío
        if not markdown_text or markdown_text.strip() == "":
            return json.dumps({"type": "doc", "content": []})
        
        # Inicializar el parser de markdown
        md = MarkdownIt()
        tokens = md.parse(markdown_text)
        
        # Estructura básica de un documento Tiptap
        tiptap_doc = {
            "type": "doc",
            "content": []
        }
        
        # Variables para mantener el estado
        current_list = None
        list_type = None
        in_code_block = False
        code_content = ""
        code_language = ""
        current_paragraph = None
        
        # Función para crear un nodo de texto
        def create_text_node(text, marks=None):
            node = {"type": "text", "text": text}
            if marks:
                node["marks"] = marks
            return node
        
        # Procesar los tokens de markdown
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Procesar encabezados
            if token.type == "heading_open":
                level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.
                content_token = tokens[i+1]
                heading_node = {
                    "type": "heading",
                    "attrs": {"level": level},
                    "content": [create_text_node(content_token.content)]
                }
                tiptap_doc["content"].append(heading_node)
                i += 3  # Saltar el token de contenido y el de cierre
                continue
            
            # Procesar párrafos
            elif token.type == "paragraph_open":
                content_token = tokens[i+1]
                
                # Procesar el contenido del párrafo para detectar formato
                paragraph_content = []
                text = content_token.content
                
                # Detectar negritas, cursivas, enlaces, etc.
                # Esto es simplificado, en un caso real necesitarías un parser más complejo
                
                # Procesar negritas (texto entre ** o __)
                bold_pattern = r'\*\*(.*?)\*\*|__(.*?)__'
                bold_matches = re.finditer(bold_pattern, text)
                
                # Procesar cursivas (texto entre * o _)
                italic_pattern = r'\*(.*?)\*|_(.*?)_'
                italic_matches = re.finditer(italic_pattern, text)
                
                # Si hay formato, procesarlo
                if re.search(bold_pattern, text) or re.search(italic_pattern, text):
                    # Aquí implementaríamos la lógica para separar el texto y aplicar los marks
                    # Por simplicidad, solo añadimos el texto completo por ahora
                    paragraph_content.append(create_text_node(text))
                else:
                    # Si no hay formato especial, añadir el texto tal cual
                    paragraph_content.append(create_text_node(text))
                
                paragraph_node = {
                    "type": "paragraph",
                    "content": paragraph_content
                }
                tiptap_doc["content"].append(paragraph_node)
                i += 3  # Saltar el token de contenido y el de cierre
                continue
            
            # Procesar listas no ordenadas
            elif token.type == "bullet_list_open":
                bullet_list = {
                    "type": "bulletList",
                    "content": []
                }
                
                # Avanzar hasta encontrar el cierre de la lista
                j = i + 1
                while j < len(tokens) and tokens[j].type != "bullet_list_close":
                    if tokens[j].type == "list_item_open":
                        # Buscar el contenido del item
                        item_content = []
                        k = j + 1
                        while k < len(tokens) and tokens[k].type != "list_item_close":
                            if tokens[k].type == "paragraph_open":
                                text = tokens[k+1].content
                                item_content.append({
                                    "type": "paragraph",
                                    "content": [create_text_node(text)]
                                })
                                k += 3  # Saltar paragraph_open, inline, paragraph_close
                            else:
                                k += 1
                        
                        # Añadir el item a la lista
                        bullet_list["content"].append({
                            "type": "listItem",
                            "content": item_content
                        })
                        
                        # Avanzar hasta después del cierre del item
                        j = k + 1
                    else:
                        j += 1
                
                tiptap_doc["content"].append(bullet_list)
                i = j + 1  # Continuar después del cierre de la lista
                continue
            
            # Procesar listas ordenadas
            elif token.type == "ordered_list_open":
                ordered_list = {
                    "type": "orderedList",
                    "content": []
                }
                
                # Avanzar hasta encontrar el cierre de la lista
                j = i + 1
                while j < len(tokens) and tokens[j].type != "ordered_list_close":
                    if tokens[j].type == "list_item_open":
                        # Buscar el contenido del item
                        item_content = []
                        k = j + 1
                        while k < len(tokens) and tokens[k].type != "list_item_close":
                            if tokens[k].type == "paragraph_open":
                                text = tokens[k+1].content
                                item_content.append({
                                    "type": "paragraph",
                                    "content": [create_text_node(text)]
                                })
                                k += 3  # Saltar paragraph_open, inline, paragraph_close
                            else:
                                k += 1
                        
                        # Añadir el item a la lista
                        ordered_list["content"].append({
                            "type": "listItem",
                            "content": item_content
                        })
                        
                        # Avanzar hasta después del cierre del item
                        j = k + 1
                    else:
                        j += 1
                
                tiptap_doc["content"].append(ordered_list)
                i = j + 1  # Continuar después del cierre de la lista
                continue
            
            # Procesar bloques de código
            elif token.type == "fence":
                code_block = {
                    "type": "codeBlock",
                    "attrs": {"language": token.info or ""},
                    "content": [create_text_node(token.content)]
                }
                tiptap_doc["content"].append(code_block)
                i += 1
                continue
            
            # Procesar líneas horizontales
            elif token.type == "hr":
                tiptap_doc["content"].append({"type": "horizontalRule"})
                i += 1
                continue
            
            # Avanzar al siguiente token si no se ha procesado
            i += 1
        
        # Asegurarse de que haya al menos un elemento de contenido
        if not tiptap_doc["content"]:
            tiptap_doc["content"].append({
                "type": "paragraph",
                "content": [create_text_node(markdown_text.strip())]
            })
        
        return json.dumps(tiptap_doc)
    
    except ImportError:
        print("Error: markdown-it-py no está instalado. Usando método alternativo.")
        # Fallback a una versión simplificada si no está disponible markdown-it-py
        tiptap_doc = {
            "type": "doc",
            "content": [{
                "type": "paragraph",
                "content": [{
                    "type": "text",
                    "text": markdown_text
                }]
            }]
        }
        return json.dumps(tiptap_doc)
    except Exception as e:
        print(f"Error al convertir markdown a Tiptap: {str(e)}")
        # En caso de error, devolver un documento simple
        tiptap_doc = {
            "type": "doc",
            "content": [{
                "type": "paragraph",
                "content": [{
                    "type": "text",
                    "text": markdown_text
                }]
            }]
        }
        return json.dumps(tiptap_doc)

def generate_brief_summary_content(text: str, output_format: str = "markdown") -> BriefSummary:
    """
    Genera un resumen breve de un texto utilizando el modelo Gemini.
    
    Esta función crea un resumen conciso que captura los puntos más importantes
    del texto en unas pocas oraciones.
    
    Args:
        text (str): El texto a resumir.
        output_format (str): Formato de salida ("markdown" o "tiptap")
        
    Returns:
        BriefSummary: Objeto con el resumen generado.
    """
    # Inicializa el modelo Gemini
    model = create_gemini_model()

    print(f"Texto proporcionado: {text[:100]}...")
    print(f"Formato de salida solicitado: {output_format}")
    
    # Si se solicita formato Tiptap, pedirlo directamente a Gemini
    if output_format.lower() == "tiptap":
        # Crea un prompt instructivo para el modelo solicitando formato Tiptap
        prompt = f"""
        Genera un resumen breve y conciso de este texto,
        capturando solo los puntos más importantes en 3-4 oraciones.
        
        IMPORTANTE: Debes devolver el resultado en formato JSON de Tiptap con esta estructura:
        {{"type": "doc", "content": [...]}}.
        
        Donde el contenido debe incluir párrafos, encabezados, listas, etc., según corresponda.
        Por ejemplo, un encabezado se representa así:
        {{"type": "heading", "attrs": {{"level": 1}}, "content": [{{"type": "text", "text": "Título"}}]}}
        
        Un párrafo se representa así:
        {{"type": "paragraph", "content": [{{"type": "text", "text": "Contenido del párrafo"}}]}}
        
        Asegúrate de que el JSON sea válido y siga la estructura de Tiptap.
        No incluyas comillas adicionales ni caracteres de escape innecesarios.
        
        Texto a resumir: {text}
        """
    else:
        # Prompt normal para formato markdown
        prompt = f"""
        Genera un resumen breve y conciso de este texto,
        capturando solo los puntos más importantes en 3-4 oraciones.

        Texto: {text}
        """

    # Obtiene la respuesta del modelo
    result = process_gemini_response(model, prompt)
    
    # Procesa el resultado según el tipo de respuesta y el formato solicitado
    if output_format.lower() == "tiptap":
        # Si se solicitó Tiptap, intentar procesar la respuesta como JSON
        try:
            if isinstance(result, str):
                # Verificar si es un JSON válido
                if result.strip().startswith('{') and 'type' in result:
                    # Parece ser JSON, devolverlo tal cual
                    return BriefSummary(summary=result)
                else:
                    # No es JSON, convertir de markdown a Tiptap
                    tiptap_json = markdown_to_tiptap_json(result)
                    return BriefSummary(summary=tiptap_json)
            elif isinstance(result, dict):
                # Si ya es un diccionario, convertirlo a JSON string
                return BriefSummary(summary=json.dumps(result))
            else:
                # Caso inesperado, convertir a string y luego a Tiptap
                tiptap_json = markdown_to_tiptap_json(str(result))
                return BriefSummary(summary=tiptap_json)
        except Exception as e:
            print(f"Error al procesar el resultado como Tiptap: {str(e)}")
            # Fallback: crear un documento Tiptap simple
            tiptap_doc = {
                "type": "doc",
                "content": [{
                    "type": "paragraph",
                    "content": [{
                        "type": "text",
                        "text": str(result)
                    }]
                }]
            }
            return BriefSummary(summary=json.dumps(tiptap_doc))
    else:
        # Formato markdown solicitado
        if isinstance(result, str):
            return BriefSummary(summary=result)
        elif isinstance(result, dict) and "summary" in result:
            return BriefSummary(summary=result["summary"])
        else:
            # Convertir cualquier otro tipo a string
            return BriefSummary(summary=str(result))

def generate_detailed_summary_content(text: str, desired_length: int = 0, output_format: str = "markdown") -> BriefSummary:
    """
    Genera un resumen detallado de un texto utilizando el modelo Gemini.
    
    Esta función crea un resumen detallado en formato de texto continuo,
    ajustándose a la longitud deseada si se especifica.
    
    Args:
        text (str): El texto a resumir en detalle.
        desired_length (int, optional): Número aproximado de palabras deseadas para el resumen.
            Si es 0, no se especifica una longitud concreta. Por defecto es 0.
        output_format (str): Formato de salida ("markdown" o "tiptap")
        
    Returns:
        BriefSummary: Objeto con el resumen detallado generado.
    """
    # Inicializa el modelo Gemini
    model = create_gemini_model()
    
    print(f"Texto proporcionado: {text[:100]}...")
    print(f"Formato de salida solicitado: {output_format}")
    print(f"Longitud deseada: {desired_length}")
    
    # Preparar instrucciones de longitud
    length_instruction = ""
    if desired_length > 0:
        length_instruction = f"""
        IMPORTANTE: El resumen DEBE tener aproximadamente {desired_length} palabras de longitud.
        No generes un resumen más corto ni mucho más largo que {desired_length} palabras.
        Esto es un requisito OBLIGATORIO. El resumen debe ser extenso y detallado, conteniendo aproximadamente {desired_length} palabras.
        No tengas en cuenta caracteres del fortmateo JSON o Markdown, solo cuenta palabras en cuanto a contenido real.
        """
    else:
        length_instruction = """
        IMPORTANTE: Hazlo lo más largo y detallado posible. Utiliza el máximo de tokens disponibles (hasta 30,000).
        No te preocupes por la extensión, queremos un resumen extremadamente detallado y completo.
        Cuanto más extenso y detallado sea, mejor. No escatimes en detalles ni en extensión.
        No tengas en cuenta caracteres del fortmateo JSON o Markdown, solo cuenta palabras en cuanto a contenido real.
        """
    
    # Si se solicita formato Tiptap, pedirlo directamente a Gemini
    if output_format.lower() == "tiptap":
        # Crea un prompt instructivo para el modelo solicitando formato Tiptap
        prompt = f"""
        Genera unos apuntes extremadamente detallados y exhaustivos de este texto.
        El apuntes debe ser un texto continuo bien estructurado que capture
        ABSOLUTAMENTE TODOS los aspectos importantes del contenido original.
        
        CRÍTICO: Hazlo lo más detallado y extenso posible, como si fueran apuntes completos para estudiar para un examen final.
        No omitas NINGÚN detalle importante. Incluye ejemplos, explicaciones, definiciones y todo lo necesario.
        
        IMPORTANTE: Debes devolver el resultado en formato JSON de Tiptap con esta estructura:
        {{"type": "doc", "content": [...]}}
        
        Donde el contenido debe incluir párrafos, encabezados, listas, etc., según corresponda.
        Por ejemplo, un encabezado se representa así:
        {{"type": "heading", "attrs": {{"level": 1}}, "content": [{{
            "type": "text", "text": "Título"
        }}]}}
        
        Un párrafo se representa así:
        {{"type": "paragraph", "content": [{{
            "type": "text", "text": "Contenido del párrafo"
        }}]}}
        
        Una lista no ordenada se representa así:
        {{"type": "bulletList", "content": [
            {{"type": "listItem", "content": [
                {{"type": "paragraph", "content": [{{
                    "type": "text", "text": "Item 1"
                }}]}}
            ]}},
            {{"type": "listItem", "content": [
                {{"type": "paragraph", "content": [{{
                    "type": "text", "text": "Item 2"
                }}]}}
            ]}}
        ]}}
        
        Asegúrate de que el JSON sea válido y siga la estructura de Tiptap.
        No incluyas comillas adicionales ni caracteres de escape innecesarios.
        
        {length_instruction}
        No pongas información del nombre del documento, título, autor...
        
        RECUERDA: Utiliza TODO el espacio disponible para crear el resumen más completo posible.
        No te preocupes por la longitud, se te permite usar hasta 30,000 tokens de salida.
        
        Texto a resumir: {text}
        """
    else:
        # Prompt normal para formato markdown
        prompt = f"""
        Genera unos apuntes detallados y completos de este texto.
        El apuntes debe ser un texto continuo bien estructurado que capture
        todos los aspectos importantes del contenido original.
        Hazlo extremadamente detallado, orientado al estudio, sin dejar detalles importantes sin tocar.
        Usa el formato markdown para estructurar, con títulos, secciones. 
        Más que un resumen, es crear unos apuntes totalmente detallados.
        Hazlo bien estructurado.
        Hazlos para estudiar para exámenes, es decir, es importante que toda la información esté ahí.
        {length_instruction}
        No pongas información del nombre del documento, título, autor...

        Texto: {text}
        """

    # Obtiene la respuesta del modelo
    result = process_gemini_response(model, prompt)
    
    # Procesa el resultado según el tipo de respuesta y el formato solicitado
    if output_format.lower() == "tiptap":
        # Si se solicitó Tiptap, intentar procesar la respuesta como JSON
        try:
            if isinstance(result, str):
                # Verificar si es un JSON válido
                if result.strip().startswith('{') and 'type' in result:
                    # Parece ser JSON, devolverlo tal cual
                    return BriefSummary(summary=result)
                else:
                    # No es JSON, convertir de markdown a Tiptap
                    tiptap_json = markdown_to_tiptap_json(result)
                    return BriefSummary(summary=tiptap_json)
            elif isinstance(result, dict):
                # Si ya es un diccionario, convertirlo a JSON string
                return BriefSummary(summary=json.dumps(result))
            else:
                # Caso inesperado, convertir a string y luego a Tiptap
                tiptap_json = markdown_to_tiptap_json(str(result))
                return BriefSummary(summary=tiptap_json)
        except Exception as e:
            print(f"Error al procesar el resultado como Tiptap: {str(e)}")
            # Fallback: crear un documento Tiptap simple
            tiptap_doc = {
                "type": "doc",
                "content": [{
                    "type": "paragraph",
                    "content": [{
                        "type": "text",
                        "text": str(result)
                    }]
                }]
            }
            return BriefSummary(summary=json.dumps(tiptap_doc))
    else:
        # Formato markdown solicitado
        if isinstance(result, str):
            return BriefSummary(summary=result)
        elif isinstance(result, dict) and "summary" in result:
            return BriefSummary(summary=result["summary"])
        else:
            # Convertir cualquier otro tipo a string
            return BriefSummary(summary=str(result))

def answer_question_with_context(question: str, collection_id: str = None, similar_documents: List[dict] = None, additional_context: str = None, conversation_history: List[dict] = None) -> dict:
    """
    Responde una pregunta utilizando documentos de contexto o Gemini directamente si no hay documentos relevantes.
        
    Args:
        question: Pregunta del usuario
        collection_id: ID de la colección de documentos (opcional)
        similar_documents: Lista de documentos similares encontrados (opcional)
        additional_context: Contexto adicional proporcionado por el usuario (opcional)
        conversation_history: Historial de la conversación (opcional)
            
    Returns:
        Dict con la respuesta y las fuentes utilizadas
    """
    try:
        import re
        
        # Detectar si es una solicitud de resumen o flashcards
        is_summary_request = False
        is_flashcard_request = False
            
        # Patrones para detectar solicitudes
        if re.search(r'(haz|genera|crea|dame|quiero|necesito)\s+(un)?\s*(resumen|resumir)', question.lower()):
            is_summary_request = True
        
        if re.search(r'(haz|genera|crea|dame|quiero|necesito)\s+(unas)?\s*(flashcards|tarjetas)', question.lower()):
            is_flashcard_request = True
        
        # Si es una solicitud de flashcards, informar al usuario sobre cómo usar la opción específica
        if is_flashcard_request:
            return {
                "answer": "No puedo generar flashcards directamente desde el chat. Por favor, utiliza el botón de 'Generar Flashcards' que se encuentra en el menú de opciones de la interfaz. Desde allí podrás seleccionar el número de flashcards y otras opciones de personalización.",
                "sources": [],
                "confidence_score": 1.0,
                "is_general_answer": True
            }
        
        # Si es una solicitud de resumen, informar al usuario sobre cómo usar la opción específica
        if is_summary_request:
            return {
                "answer": "No puedo generar resúmenes directamente desde el chat. Por favor, utiliza el botón de 'Resumen Breve/Detallado' que se encuentra en el menú de opciones de la interfaz. Desde allí podrás elegir entre un resumen detallado o breve según tus necesidades.",
                "sources": [],
                "confidence_score": 1.0,
                "is_general_answer": True
            }
        
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Si no hay documentos similares o están vacíos, usar Gemini directamente
        if not similar_documents or len(similar_documents) == 0:
            prompt = f"""
            Por favor, responde a la siguiente pregunta de manera clara y concisa: {question}

            Instrucciones:
            1. Proporciona una respuesta directa y útil
            2. Si la pregunta requiere información específica que no tienes, indícalo claramente
            3. Mantén un tono profesional y objetivo
            """
                
            response = model.generate_content(prompt)
            return {
                "answer": response.text,
                "sources": [],
                "confidence_score": 0.7,  # Confianza moderada para respuestas sin contexto específico
                "is_general_answer": True  # Indicador de que es una respuesta general
            }
        
        # Si hay documentos, proceder con el flujo normal
        context = "Basándome en los siguientes documentos:\n\n"
        sources = []
        
        # Filtrar documentos por relevancia (usando un umbral de similitud)
        similarity_threshold = 0.5  # Ajustar según sea necesario
        filtered_documents = [doc for doc in similar_documents if doc.get("similarity_score", 0) >= similarity_threshold]
        
        for doc in filtered_documents:
            # Preparar la información del documento
            doc_info = {
                "document_id": doc["document_id"],
                "content": doc["content"],
                "file_name": doc["file_name"],
                "file_type": doc["file_type"],
                "similarity_score": doc["similarity_score"],
                "relevant_lines": []
            }
            
            # Agrupar líneas por página (asumiendo 40 líneas por página)
            page_size = 40
            lines_by_page = {}
            
            for line_info in doc.get("relevant_lines", []):
                line_number = line_info["line_number"]
                page_number = (line_number - 1) // page_size + 1
                relative_line = (line_number - 1) % page_size + 1
                
                if page_number not in lines_by_page:
                    lines_by_page[page_number] = []
                
                # Crear una nueva estructura para la línea con número de línea relativo a la página
                lines_by_page[page_number].append({
                    "line_number": line_number,  # Número de línea absoluto (para compatibilidad)
                    "page_line_number": relative_line,  # Número de línea relativo a la página
                    "content": line_info["content"]
                })
            
            # Ordenar las páginas y las líneas dentro de cada página
            sorted_pages = sorted(lines_by_page.keys())
            
            # Añadir al contexto solo si hay líneas relevantes
            if sorted_pages:
                context += f"\nDocumento: {doc_info['file_name']} ({doc_info['file_type']})\n"
                
                for page in sorted_pages:
                    context += f"\nPágina {page}:\n"
                    
                    # Ordenar líneas por número de línea relativo
                    sorted_lines = sorted(lines_by_page[page], key=lambda x: x["page_line_number"])
                    
                    for line in sorted_lines:
                        context += f"  Línea {line['page_line_number']}: {line['content']}\n"
                        
                        # Guardar la línea en el formato esperado por el frontend
                        doc_info["relevant_lines"].append({
                            "line_number": line["line_number"],  # Mantener el número absoluto para compatibilidad
                            "content": line["content"]
                        })
                
                # Añadir el documento a las fuentes independientemente de si tiene líneas relevantes
                sources.append(doc_info)
            else:
                # Si no hay líneas relevantes pero hay contenido, mostrar el contenido completo
                context += f"Contenido: {doc_info['content']}\n"
                # También añadir el documento a las fuentes
                sources.append(doc_info)

        # Añadir contexto adicional si está disponible
        additional_context_text = ""
        if additional_context:
            additional_context_text = f"\nContexto adicional proporcionado por el usuario:\n{additional_context}\n"
        
        # Añadir historial de conversación si está disponible
        conversation_history_text = ""
        previous_context = ""
        
        if conversation_history and len(conversation_history) > 0:
            # Extraer el contexto de las preguntas y respuestas anteriores
            conversation_history_text = "\nHistorial de la conversación:\n"
            
            # Analizar el historial para extraer temas y contexto relevante
            topics = []
            last_question = ""
            last_answer = ""
            
            for i, msg in enumerate(conversation_history):
                # Acceder a los atributos directamente en lugar de usar get()
                role = "Usuario" if msg.role == "user" else "Asistente"
                content = msg.content if hasattr(msg, 'content') else ""
                conversation_history_text += f"{role}: {content}\n"
                
                # Guardar la última pregunta y respuesta para contexto
                if i > 0 and i == len(conversation_history) - 2 and role == "Usuario":
                    last_question = content
                elif i > 0 and i == len(conversation_history) - 1 and role == "Asistente":
                    last_answer = content
                
                # Extraer posibles temas del contenido
                if content:
                    # Buscar palabras clave o términos técnicos en mayúsculas
                    import re
                    keywords = re.findall(r'\b[A-Z]{2,}\b', content)
                    topics.extend(keywords)
            
            # Crear contexto para la pregunta actual basado en el historial
            if last_question and last_answer:
                previous_context = f"\nContexto de la conversación anterior:\n"
                previous_context += f"La pregunta anterior fue sobre: {last_question}\n"
                previous_context += f"La respuesta proporcionada fue: {last_answer}\n"
                
                if topics:
                    previous_context += f"Temas relevantes mencionados: {', '.join(set(topics))}\n"
                
                previous_context += "\nLa pregunta actual debe interpretarse en el contexto de esta conversación.\n"
            
            conversation_history_text += "\n"
            
        prompt = f"""{context}{additional_context_text}{previous_context}{conversation_history_text}

Por favor, responde la siguiente pregunta teniendo en cuenta el contexto de la conversación: {question}

Instrucciones importantes:
1. Si la pregunta actual hace referencia a una pregunta o respuesta anterior, asegúrate de interpretar correctamente el contexto.
2. Si la pregunta es una continuación o está relacionada con el tema anterior, mantén la coherencia en tu respuesta.
3. Si la pregunta es "¿Es algo que funciona bien?" o similar, asume que se refiere al tema principal de la conversación anterior.
4. Si la respuesta no se encuentra en los documentos proporcionados, indica que no tienes esa información.
5. Si la respuesta se encuentra en los documentos, cita las fuentes relevantes al final de tu respuesta.

    Instrucciones adicionales:
    1. Basa tu respuesta en la información proporcionada en los documentos, pero preséntala de forma natural y fluida.
    2. IMPORTANTE: No menciones explícitamente frases como "Según el documento X..." o "Como se indica en el documento Y...". 
       En su lugar, integra la información de manera natural en tu respuesta.
    3. Puedes añadir contexto adicional y ampliar la información para que la respuesta sea más completa y útil.
    4. Mantén un tono profesional, claro y directo.
    5. Si la información no está disponible en los documentos, indícalo de forma clara pero sin mencionar los documentos.
    6. Si la pregunta contiene términos técnicos o acrónimos, asegúrate de explicarlos adecuadamente.
    7. Proporciona ejemplos cuando sea apropiado, incluso si no están explícitamente en los documentos.
    8. Recuerda que el objetivo es proporcionar una respuesta útil y educativa, no simplemente extraer información de los documentos.
    9. IMPORTANTE: Si hay un historial de conversación, ten en cuenta el contexto de las preguntas y respuestas anteriores para proporcionar una respuesta coherente y contextualizada.
    """

        response = model.generate_content(prompt)
        confidence_score = max([doc.get("similarity_score", 0) for doc in filtered_documents]) if filtered_documents else 0.0

        return {
            "answer": response.text,
            "sources": sources,
            "confidence_score": confidence_score,
            "is_general_answer": False
        }

    except Exception as e:
        print(f"Error al responder la pregunta: {str(e)}")
        raise e

def answer_general_question_content(question: str) -> QuestionAnswer:
    """
    Responde una pregunta general sin contexto específico utilizando el modelo Gemini.
    
    Esta función utiliza el conocimiento general del modelo para responder preguntas
    sin necesidad de documentos de contexto específicos.
    
    Args:
        question (str): La pregunta general a responder.
        
    Returns:
        QuestionAnswer: Objeto con la respuesta y metadatos.
    """
    # Inicializa el modelo Gemini
    model = create_gemini_model()
    
    # Crea un prompt instructivo para el modelo
    prompt = f"""
    Responde esta pregunta general usando tu conocimiento.
    Si no tienes suficiente información, indícalo claramente.

    Pregunta: {question}
    """

    # Obtiene la respuesta del modelo
    result = process_gemini_response(model, prompt)
    
    # Convierte el resultado en un objeto QuestionAnswer
    return QuestionAnswer(**result)

# Función para buscar documentos relevantes (stub)
def find_relevant_documents(question: str, collection_id: str, limit: int = 5) -> List[str]:
    """
    Encuentra documentos relevantes para una pregunta usando similitud de embeddings.
    
    Esta es una implementación de ejemplo. En una aplicación real, esta función
    buscaría en una base de datos de vectores utilizando el embedding de la pregunta.
    
    Args:
        question (str): La pregunta para la cual buscar documentos relevantes.
        collection_id (str): ID de la colección donde buscar.
        limit (int, optional): Número máximo de documentos a devolver. Default: 5.
        
    Returns:
        List[str]: Lista de documentos relevantes encontrados.
    """
    # En una implementación real, aquí se buscaría en una base de datos vectorial
    # Por ahora, devolvemos una lista vacía
    return []

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extrae texto de un archivo PDF utilizando la biblioteca PyPDF2.
    
    Esta función abre el archivo PDF, lee su contenido y devuelve el texto extraído.
    
    Args:
        file_path (str): Ruta al archivo PDF.
        
    Returns:
        str: Texto extraído del archivo PDF.
    """
    text = ''
    with open(file_path, 'rb') as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        num_pages = len(pdf_reader.pages)
        for page in range(num_pages):
            page_obj = pdf_reader.pages[page]
            text += page_obj.extract_text()
    return text

def extract_text_from_pdf_base64(pdf_base64: str) -> str:
    """
    Extrae texto de un archivo PDF codificado en base64 utilizando la biblioteca PyPDF2.
    
    Esta función decodifica el archivo PDF, lee su contenido y devuelve el texto extraído.
    
    Args:
        pdf_base64 (str): Archivo PDF codificado en base64.
        
    Returns:
        str: Texto extraído del archivo PDF.
    """
    pdf_bytes = base64.b64decode(pdf_base64)
    pdf_file_obj = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    num_pages = len(pdf_reader.pages)
    text = ''
    for page in range(num_pages):
        page_obj = pdf_reader.pages[page]
        text += page_obj.extract_text()
    return text
