# **Proyecto final - IA Generativa con LLM**
***
## Bienvenido a PsyGenius AI.
### Sistema de evaluación psicológica con IA Generativa.
<br />
<br />
Soy el alumno **José Llashag Huamán**. 
PsyGenius AI es un sistema de evaluación psicológica diseñado para empleados de empresas.
<br />
<br />
### Antes de ejecutar el proyecto:
<br />
**1.** En el archivo "cargafiles.ipynb" se encuentra el código que se utilizó para alimentar la Base de Datos de Vectores. 
Se cargó el archivo "pdf/Ronald_E_Riggio_Introduction_to_Industrial_Orga.pdf" en Pinecone con el objetivo de que el modelo aprenda a formular preguntas para evaluar la satisfacción laboral del empleado.
<br />
**2.** En el archivo ".env" se encuentran los Keys, Endpoint y otros parámetros necesarios para el proyecto:
  - **PINECONE_API_KEY:** Key de Pinecone.
  - **OPENAI_API_KEY:** Key de OpenAI.
  - **INDEX_NAME:** Nombre del Index en Pinecone donde se almacenan los vectores.
  - **CANT_PREGUNTAS:** Parámetro sobre la cantidad de preguntas que el sistema puede realizar al empleado. Por defecto es 5.
  - **SPEECH_KEY y SPEECH_REGION:** Key y Region de Speech Services de Azure.
  - **AZURE_LANGUAGE_KEY y AZURE_LANGUAGE_ENDPOINT:** Key y Endpoint de Language Services de Azure.
  - **AI_SERVICE_KEY y AI_SERVICE_ENDPOINT:** Key y Endpoint de Cognitive Services de Azure.
<br />
**3.** Ejecutar el archivo "requirements.txt" para instalar todas las librerías necesarias.
<br />
**4.** Ejecutar el archivo "app.py" para levantar el proyecto.
<br />
<br />
### Pasos a seguir:
<br />
**1.** Hacer clic en el botón "Ver video" (opcional). Es un video con una introducción breve.
<br />
**2.** Hacer clic en el botón "Iniciar evaluación" para empezar. Escuchar atentamente los pasos que debe seguir.
<br />
**3.** En la 1ra parte de la evaluación, debe escuchar atentamente las preguntas y responder cada una de ellas. Cuando termine de responder una pregunta, debe decir la palabra "Siguiente Pregunta", para que el sistema continúe con la siguiente.
<br />
**4.** Una vez que termina de responder todas las preguntas, hacer clic en el botón "Guardar"
<br />
**5.** En la 2da parte, escriba una pequeña historia de libre elección. Al terminar, hacer clic en Finalizar.
<br />
**6.** El sistema descargará un PDF con los resultados. Debe ser paciente, la descarga puede demorar un poco.
<br />
Si bien el PDF debe ser enviado al área de RRHH de la empresa, para efectos prácticos se optó que sea descargado en el navegador automáticamente.
<br />
<br />
### Recomendaciones:
<br />
**1.** Una buena velocidad de Internet ayudará que los servicios de las APIs tengan una respuesta más rápida.
<br />
**2.** Tener una buena iluminación, ya que la cámara tomará una foto para detectar su rostro.
<br />
**3.** Responder claramente las preguntas para que el servicio Speech Services de Azure reconozca sus palabras.
<br />
**4.** Ser paciente, el sistema puede demorar en brindale los pasos que debe seguir.
<br />
**5.** Puede visualizar en el Terminal los "print" que se van ejecutando, de esta manera tendrá un mejor seguimiento en que proceso se encuentra la aplicación.
