from flask import Flask, request, render_template, Response 
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from fpdf import FPDF
import azure.cognitiveservices.speech as speech_sdk
import os
import re
import typing
import pinecone

app = Flask(__name__)

cant_preguntas = 0
dict_final = {}
cant_face = 0

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/inicio', methods=['POST'])
def inicio():
    return render_template("index.html")

@app.route('/evaluacion', methods=['POST'])
def evaluacion():
    response_text = 'Vamos a iniciar con la evaluación psicológica. Por favor, haga clic en Iniciar Evaluación y espere con paciencia.'
    ejecutaraudio(response_text)
    return ('', 204)

@app.route('/start_evaluacion', methods=['POST'])
def start_evaluacion():
    global dict_final, cant_preguntas
    cant_preguntas = os.getenv('CANT_PREGUNTAS')
    
    print("Consultando Pinecone...")
    myList = consultar_pinecone()
    print(myList)
    dict_final["Preguntas"] = myList
    
    response_text = "Responde brevemente las siguientes preguntas. Cuando quieras pasar a la siguiente pregunta, menciona la palabra: Siguiente Pregunta"
    ejecutaraudio(response_text)

    count = 0
    array_resp = []
    
    for queslist in myList:
        if queslist != "No lo sé, formula bien tu pregunta.":
            array_resp.append("")
            
            ejecutaraudio(queslist)
            palabranext = False
            
            while not palabranext:
                command = escucharaudio()
                array_resp[count] = str(array_resp[count]) + command + " "
                
                if 'siguiente pregunta' in command.lower():
                    palabranext = True
                    print("Siguiente pregunta")
                    
            count += 1
    
    dict_final["Respuestas"] = array_resp
    
    response_text = "Ha respondido " + cant_preguntas + " preguntas con éxito. Favor de hacer clic en el botón Guardar"
    ejecutaraudio(response_text)
    
    return 'Terminando primera evaluación.'

@app.route('/save_evaluacion', methods=['POST'])
def save_evaluacion():
    foto_file = request.files["foto"]
    foto_path = os.path.join("files", "Foto_PsyGenius.png")
    foto_file.save(foto_path)

    response_text = "Ahora vamos a pasar al siguiente nivel. Haga clic en Escritura Creativa y escriba una pequeña historia"
    ejecutaraudio(response_text)
    
    return 'Video guardado con éxito.'

@app.route('/finalizar', methods=['POST'])
def finalizar():
    features = [x for x in request.form.values()]
    texto = features[0]
    dict_final["Escritura"] = texto
    
    print("Analizando texto...")
    textfinal = analyze_sentiment_with_opinion_mining(texto)
    dict_final["Analisis"] = textfinal
    
    print("Analizando rostro...")
    analyze_face()
    
    print(dict_final)
    
    response_text = "Finalizó con éxito la evaluación. En breve se descargará un PDF."
    ejecutaraudio(response_text)
    print("Generando PDF...")
    
    return generar_pdf()

def consultar_pinecone():
    load_dotenv()
    api_key = os.getenv('PINECONE_API_KEY')
    os.environ["PINECONE_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
    index_name = os.getenv('INDEX_NAME')
    text_field = "text"
    
    pc = pinecone.Pinecone(api_key = api_key)
    index = pc.Index(index_name)
    embed = OpenAIEmbeddings()
    vectorstore = Pinecone(index, embed.embed_query, text_field)
    
    template = """Responda a la pregunta basada en el siguiente contexto.
    Si no puedes responder a la pregunta, usa la siguiente respuesta "No lo sé, formula bien tu pregunta."

    Contexto: 
    {context}
    Pregunta: {question}
    Respuesta: 
    """

    prompt = PromptTemplate(template = template, input_variables = ["context", "question"])
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0.0)

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorstore.as_retriever(),
        verbose = False,
        chain_type_kwargs = {"prompt": prompt, "verbose": False}
    )
    
    text_quest = "realiza " + cant_preguntas + " preguntas para saber la satisfacción laboral del empleado"
    text_resp = qa.run(text_quest)
    
    ### En caso la respuesta sea: "No lo sé, formula bien tu pregunta." ###
    #Funciona bien la consulta a Pinecone, sin embargo, a veces no responde la pregunta realizada. Es un error que no he podido solucionar aún
    #No obstante, en la mayoría de las consultas a Pinecone se obtiene el resultado esperado
    if text_resp == "No lo sé, formula bien tu pregunta.":
        print("Problemas con la consulta... Plan B...")
        text_resp = "1. ¿Cómo calificaría su nivel de satisfacción con su trabajo actual?\n2. ¿Qué aspectos específicos de su trabajo le generan mayor satisfacción?\n3. ¿Qué cambios o mejoras le gustaría ver en su trabajo para aumentar su satisfacción?\n4. ¿Cómo describiría su relación con sus compañeros de trabajo y supervisores en términos de satisfacción?\n5. ¿Qué actividades o tareas adicionales realiza en su trabajo que contribuyen a su satisfacción laboral?"
    else:
        print("Consulta a Pinecone exitoso...")
    
    myList = re.split('\n', text_resp)
    
    return myList

def ejecutaraudio(response_text):
    try:
        global speech_config
        load_dotenv()
        
        ai_key = os.getenv('SPEECH_KEY')
        ai_region = os.getenv('SPEECH_REGION')
        language = 'es-ES'
        
        speech_config = speech_sdk.SpeechConfig(ai_key, ai_region, speech_recognition_language=language)        
        speech_config.speech_synthesis_voice_name = "es-PE-CamilaNeural"
        
        speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config)
        speak = speech_synthesizer.speak_text_async(response_text).get()
        
        if speak.reason != speech_sdk.ResultReason.SynthesizingAudioCompleted:
            print(speak.reason)

    except Exception as ex:
        print(ex)

def escucharaudio():
    command = ''
    audio_config = speech_sdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)
    speech = speech_recognizer.recognize_once_async().get()
    
    if speech.reason == speech_sdk.ResultReason.RecognizedSpeech:
        command = speech.text
        print(command)
    else:
        print(speech.reason)
    if speech.reason == speech_sdk.ResultReason.Canceled:
        cancellation = speech.cancellation_details
        print(cancellation.reason)
        print(cancellation.error_details)

    return command

def analyze_sentiment_with_opinion_mining(texto):
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_KEY"]
    textfinal = []

    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    documents = [texto]

    result = text_analytics_client.analyze_sentiment(documents, show_opinion_mining=True)
    doc_result = [doc for doc in result if not doc.is_error]

    positive_reviews = [doc for doc in doc_result if doc.sentiment == "positive"]
    mixed_reviews = [doc for doc in doc_result if doc.sentiment == "mixed"]
    negative_reviews = [doc for doc in doc_result if doc.sentiment == "negative"]
    
    textfinal.append("Hay " + str(len(positive_reviews)) + " opinion(es) positiva(s), " + str(len(mixed_reviews)) + " opinion(es) mixta(s) y "+ str(len(negative_reviews)) + " opinion(es) negativa(s). ")
    
    target_to_complaints: typing.Dict[str, typing.Any] = {}

    for document in doc_result:
        for sentence in document.sentences:
            if sentence.mined_opinions:
                for mined_opinion in sentence.mined_opinions:
                    target = mined_opinion.target
                    target_to_complaints.setdefault(target.text, [])
                    target_to_complaints[target.text].append(mined_opinion)

    for target_name, complaints in target_to_complaints.items():
        texteval = "El empleado ha escrito " + str(len(complaints)) + " aspecto(s) clave(s). Acerca de '" + target_name + "', menciona que es '" + "', '".join([assessment.text for complaint in complaints for assessment in complaint.assessments]) + "'"
        textfinal.append(texteval)
        
    return textfinal

def analyze_face():
    global face_client, cant_face

    try:
        load_dotenv()
        cog_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        cog_key = os.getenv('AI_SERVICE_KEY')

        credentials = CognitiveServicesCredentials(cog_key)
        face_client = FaceClient(cog_endpoint, credentials)

        image_file = os.path.join('files','Foto_PsyGenius.png')
        print('Detecting faces in', image_file)

        ### Algunos atributos fueron retirados por Microsoft
        #https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/identity-detect-faces
        #"Microsoft ha retirado las capacidades de reconocimiento facial que pueden usarse para tratar de inferir 
        #estados emocionales y atributos de identidad que, si se usan mal, pueden someter a las personas a estereotipos, 
        #discriminación o denegación injusta de servicios. Estos incluyen capacidades que predicen 
        #emociones, género, edad, sonrisa, vello facial, cabello y maquillaje."
        features = [#FaceAttributeType.age, #Retirado por Microsoft
                    #FaceAttributeType.emotion, #Retirado por Microsoft
                    #FaceAttributeType.gender, #Retirado por Microsoft
                    #FaceAttributeType.smile #Retirado por Microsoft
                    FaceAttributeType.head_pose,
                    FaceAttributeType.accessories,
                    FaceAttributeType.glasses,
                    FaceAttributeType.exposure
                    ]

        with open(image_file, mode="rb") as image_data:
            detected_faces = face_client.face.detect_with_stream(image=image_data, return_face_attributes=features, return_face_id=False)

        cant_face = len(detected_faces)
        
        if len(detected_faces) > 0:
            print(len(detected_faces), 'faces detected.')
            
            fig = plt.figure(figsize=(16, 14))
            plt.axis('off')
            image = Image.open(image_file)
            draw = ImageDraw.Draw(image)
            color = 'lightgreen'
            face_count = 0
            array_face = []

            for face in detected_faces:
                face_count += 1
                detected_attributes = face.face_attributes.as_dict()
                
                if 'quality_for_recognition' in detected_attributes:
                    array_face.append('Calidad de reconocimiento: ' + str(detected_attributes['quality_for_recognition']))

                if 'head_pose' in detected_attributes:
                    array_face.append('Postura de la cabeza: ' + str(detected_attributes['head_pose']))
                
                if 'accessories' in detected_attributes:
                    array_face.append('Accesorios: ' + str(detected_attributes['accessories']))
                
                if 'glasses' in detected_attributes:
                    array_face.append('Lentes: ' + str(detected_attributes['glasses']))
                
                if 'exposure' in detected_attributes:
                    array_face.append('Exposición: ' + str(detected_attributes['exposure']))
                
                r = face.face_rectangle
                bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
                draw = ImageDraw.Draw(image)
                draw.rectangle(bounding_box, outline=color, width=5)
                plt.annotate("", (r.left, r.top), backgroundcolor=color)
                
            plt.imshow(image)
            outputfile = os.path.join('files','Face_PsyGenius.png')
            fig.savefig(outputfile)
            
            dict_final["Rostros"] = array_face
            
        else:
            print('No se detecta rostros')

    except Exception as ex:
        print(ex)

def generar_pdf():
    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Resultado final de la evaluación", ln=1, align="C")
    
    pdf.set_font("Arial", "", 16)
    pdf.cell(0, 10, "\n", ln=1, align="C")
    
    pdf.set_font("Arial", "", 16)
    pdf.cell(0, 10, "\n", ln=1, align="C")

    for key, value in dict_final.items():        
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 5, key + ":")
        
        pdf.set_font("Arial", "", 16)
        pdf.cell(0, 10, "\n", ln=1, align="C")
        
        if key == "Preguntas":
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 5, "Se realizaron " + str(cant_preguntas) + " preguntas.")
            pdf.set_font("Arial", "", 16)
            pdf.cell(0, 10, "\n", ln=1, align="C")
        
        if key == "Rostros":
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 5, "Se detectaron " + str(cant_face) + " rostro(s).")
            pdf.set_font("Arial", "", 16)
            pdf.cell(0, 10, "\n", ln=1, align="C")
            
        if key != "Escritura":
            for valor in value:
                if key != "Respuestas":
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 5, valor)
                else:
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 5, replacevalores(valor))
        else:
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 5, value)
            
        pdf.set_font("Arial", "", 16)
        pdf.cell(0, 10, "\n", ln=1, align="C")

    if os.path.exists(os.path.join('files','Face_PsyGenius.png')):
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 5, "Imagen del rostro detectado: ")
        
        getx = pdf.get_x()
        gety = pdf.get_y()
        pdf.image(os.path.join('files','Face_PsyGenius.png'), x=getx, y=gety, w=200, h=150)
    
    response = Response(pdf.output(dest='S').encode('latin-1'),
                        mimetype='application/pdf',
                        headers={'Content-Disposition': 'attachment; filename=EvaluacionFinal.pdf'})
    
    return response

def replacevalores(valor):
    val = valor.replace("siguiente pregunta.", "").replace("Siguiente pregunta.", "").replace(".", "").replace(",", "")
    val = val.strip() + "."
    return val

if __name__ == '__main__':
    app.run(debug=True)
