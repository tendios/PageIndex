# Búsqueda conjunta opcional

`MultiDocumentSearch` es una API experimental del fork desde `v0.5.0-tendios.3`.
Se activa por una llamada distinta. `PageIndexClient.chat()` conserva sus prompts,
herramientas, comportamiento y formatos, tanto con uno como con varios documentos.
Los índices y los árboles almacenados no cambian.

## Qué modo elegir

| Necesidad | Entrada |
|---|---|
| Un documento, chat habitual, streaming o conversación con historial | `client.chat(...)` |
| Varios documentos cuyos árboles y lecturas caben cómodamente en el contexto | `client.chat(..., doc_id=[...])` sigue siendo el punto de partida |
| Revisión conjunta de estructuras grandes, con presupuestos y registro explícito de trabajo pendiente | `MultiDocumentSearch(...).query(...)`, evaluando primero calidad, latencia y coste |
| Búsqueda de evidencias por documento o validación para el comparador | Los flujos de evidencias existentes del service; no se conectan automáticamente a esta API |

El número de documentos por sí solo no elige el modo. El llamador o service puede
exponer `search_mode="standard" | "multi_document"`, con `standard` por defecto.
Ese parámetro pertenece al service: **no es un argumento de `client.chat()`**.
No hay enrutamiento ni fallback automático dentro de la librería.

La revisión de todos los árboles cuesta llamadas: no es una optimización de
velocidad garantizada. La mejora buscada es controlar el contexto y declarar qué
se examinó, sin eliminar documentos al alcanzar un presupuesto. La calidad debe
compararse con la búsqueda actual sobre el corpus de cada aplicación.

## Uso

```python
from pageindex import PageIndexClient, MultiDocumentSearch, MultiDocumentOptions

client = PageIndexClient(
    index={"store": {"type": "file", "path": ".pageindex"}},
    chat="anthropic/claude-haiku-5-5",
)
search = MultiDocumentSearch(client, options=MultiDocumentOptions(
    context_window=32768,        # configurar para el endpoint real
    max_input_tokens=12000,      # estimación con margen, por llamada
    max_output_tokens=4096,
    safety_tokens=1024,
    max_calls=64,
    max_total_tokens=200000,     # reservas conservadoras de entrada + salida
    timeout_seconds=180,
))
result = search.query(
    "Compara los plazos, incluyendo excepciones y discrepancias.",
    doc_ids=["contrato", "anexo"],
)
if result["status"] == "complete":
    print(result["answer"])
else:
    print(result["status"], result["reasons"], result["pending_stage"])
```

Requiere dos IDs distintos, índice local y un modelo API accesible mediante
LiteLLM. Hereda `client.chat_model`, `client.chat_backend` y las instrucciones del
cliente; admite `model=` y `backend=` al construir la instancia, e `instructions=`
por consulta. Rechaza cloud, `claude-code/` y opciones de conexión que sobrescriban
el contenido o los presupuestos controlados por esta API. No admite streaming ni
historial conversacional en esta primera versión. La instancia puede compartirse
entre consultas: el estado y las conexiones parametrizadas son propios de cada
ejecución; no cambia atributos del cliente.

`max_total_tokens` controla una **reserva conservadora**: cada intento consume su
entrada estimada más la salida máxima, aunque genere menos. `usage` contiene los
tokens que informa el proveedor; no confundir ambos. Si falta usage, la reserva
se mantiene y `usage_complete=false` lo indica. Si una respuesta estructurada se
corta por longitud, el lote se divide en dos cuando es posible; todos los intentos
cuentan y aparecen en `calls`, con `recovery="split_batch"`. No hay reintentos
ocultos del transporte ni caché persistente de respuestas.

Los presupuestos de contexto son estimaciones del tokenizador con un margen
configurable (`token_margin=0.60`) y reserva adicional. No son una garantía del
conteo de todos los proveedores; comprobar las métricas reales del endpoint.
El deadline limita solicitudes al modelo; las lecturas síncronas del almacén y
la comprobación final de versiones no se pueden interrumpir por ese timeout.

## Proceso y calidad

1. Obtiene con APIs públicas metadatos, árboles y páginas almacenadas, y calcula
   huellas de contenido. También identifica índices antiguos sin `_revision`.
2. Examina todos los registros del árbol en lotes deterministas con un contexto
   nuevo por llamada. Reparte registros entre documentos. Cada sección indica
   cuántas subsecciones la siguen (`children`), para que la selección se haga en
   el nivel más específico en vez de leer rangos completos. Un registro enorme se
   fragmenta sin perder su texto ni identidad.
3. El modelo selecciona páginas según la pregunta. Leer un resumen o un árbol
   **no cuenta como leer las páginas**. La selección sigue siendo una decisión
   del modelo y puede omitir una fuente pertinente.
4. Presenta el texto original seleccionado en lotes, fragmentando páginas grandes
   con solapamiento. Conserva hallazgos y fuentes fuera del contexto del modelo.
   Las citas se comprueban literalmente con normalización Unicode y de espacios.
5. Sintetiza los hallazgos citando sus identificadores. Resuelve cajas con el
   mismo localizador del SDK, sobre la instantánea de páginas, y comprueba de
   nuevo las huellas antes de devolver fuentes utilizables.

Los árboles no se fusionan destruyendo la procedencia. Cada sección conserva
documento, padre, nodo y páginas. No se crea un índice temático automáticamente.
La geometría no entra en los prompts. La instantánea local incluye páginas y
layout: implica memoria e I/O adicionales frente a la consulta habitual.

**Cuando los hallazgos no caben juntos**, devuelve síntesis por lotes en
`answer_parts` y estado `partial` con `synthesis_partitioned`; `answer` permanece
vacío. No encadena resúmenes y presenta el resultado como una comparación global
completa. Un llamador puede mostrar las partes como parciales, o repetir con
mayor presupuesto/contexto. No hay continuación automática entre llamadas.

`complete` significa que terminó ese recorrido, no que se leyó cada página,
que no falte información, ni que la interpretación del modelo sea correcta.
La validación literal confirma el texto almacenado; no valida semánticamente la
conclusión ni garantiza que el OCR reconozca correctamente la imagen.

## Contrato `pageindex.multidoc/1`

| Campo | Significado |
|---|---|
| `status` | `complete`, `partial` o `failed` |
| `answer` | Respuesta global con referencias `[E000001]`; vacío si no pudo sintetizarse o cambió una fuente |
| `answer_parts` | Respuestas por lote, con `finding_ids`; no tratarlas como comparación global cuando el estado es parcial |
| `citations` | Hallazgos literales: `id`, `doc_id`, `name`, `page`, `quote`, `note`, `source_id`, `fingerprint`, `locations`, `geometry_status`, `match`, `validation` |
| `documents` | Huella/estado por documento, partes estructurales totales/examinadas/pendientes, páginas seleccionadas/leídas/pendientes, `extraction_pending` y `all_pages_read` |
| `reasons`, `pending_stage` | Causa y etapa no completada |
| `usage`, `usage_complete` | Suma de tokens informados y si todos los intentos informaron consumo |
| `budget_tokens_reserved` | Reserva conservadora consumida, distinta del consumo facturable real |
| `calls` | Etapa, estimación de entrada, consumo, terminación, milisegundos y recuperación si hubo división |
| `model`, `limits`, `ms`, `notes` | Modelo, presupuestos y limitaciones de interpretación |

`citations` conserva todos los hallazgos validados, incluidos los no citados en
la respuesta final. Para mostrar solo los citados, `pageindex.multidoc.cited_ids(answer)`
devuelve los ids citados en orden (tolera anotaciones dentro del corchete, como
`[E000001, pág. 6]`); cruzarlos con `citations[].id`. `note` es interpretación del
modelo. `quote` es texto literal.
No pasar directamente el nuevo `answer` a `client.get_citations()`: su formato de
referencias es distinto del `<cite>` del chat habitual.

`locations` devuelve todas las coincidencias de esa cita en su página: no elige
una caja arbitraria cuando hay varias. Su formato es el de
`client.get_text_locations()`: cajas normalizadas 0–1000, origen superior izquierdo,
y cajas en puntos. Sin coincidencia geométrica sigue existiendo una cita literal
válida en el texto de página. `geometry_status` distingue `available`, `no_match`
y `unavailable` (error de geometría).

Si una fuente cambia o desaparece, el estado es parcial, se vacían la respuesta y
sus partes, se retiran las citas del documento afectado y este se marca `stale`.
La verificación es optimista, no una transacción que bloquee reindexaciones.
Un estado `current` solo describe la comprobación al final de esa ejecución.

`pages_read` exige que todos los fragmentos de la página se hayan enviado a un
modelo que devolvió respuesta. Incluye respuestas cortadas o inválidas;
`extraction_pending` indica qué pasajes no completaron la extracción validada.

Causas actuales: `max_calls`, `max_total_tokens`, `timeout`, `input_budget`,
`incomplete_answer`, `invalid_response`, `invalid_reference`, `invalid_quote`,
`model_error`, `source_error`, `source_unavailable`, `source_changed`,
`document_not_ready`, `synthesis_partitioned`. No convertirlas en respuestas
que afirmen ausencia de información.

`include_trace=True` añade prompts y respuestas a `calls`. Puede contener el
texto de documentos: no activarlo ni registrarlo indiscriminadamente en producción.

## Inspección y ampliación de ramas sin IA

```python
page = search.structure_page(doc_ids=["contrato", "anexo"])
while page["has_more"]:
    page = search.structure_page(
        doc_ids=["contrato", "anexo"], cursor=page["next_cursor"],
    )

branch = search.structure_page(
    doc_ids=["contrato", "anexo"],
    root={"doc_id": "contrato", "node_id": "0012"},
)
```

Devuelve registros, paginación, huellas, estimación de tokens del árbol completo
y presupuesto de entrada. Los cursores dependen del orden/alcance, la pregunta,
la rama, los presupuestos y las versiones; si cambian se rechazan. No cuentan
como cobertura de una consulta. Un fragmento serializado lleva offsets para
reconstruir el registro íntegro. El caller debe conservar el mismo `question`
al continuar si lo especificó en la primera llamada.

## Benchmark reproducible

```bash
# Solo prepara el corpus sintético y el manifiesto; no llama modelos.
python cookbook/multidoc/benchmark.py --output results/multidoc-benchmark

# Llamadas de pago explícitas. Mismo modelo y fuentes en ambos modos.
python cookbook/multidoc/benchmark.py --output results/multidoc-benchmark \
  --run --repeats 2 --model anthropic/claude-haiku-5-5
```

Diez contratos sintéticos, 300 páginas, respuestas conocidas: documento décimo,
excepción profunda, comparación, tabla completa, cita textual, combinación de
condiciones, contradicción y ausencia de información. El benchmark reutiliza
índices controlados, sin medir OCR ni calidad del indexador. Guarda manifiesto,
huella del corpus, respuestas, consumo, latencias y trazas en JSONL.

Los checks automáticos de palabras/números y páginas esperadas son indicadores
limitados, no un juez semántico. Revisar manualmente omisiones, atribuciones,
negaciones y condiciones. Las conclusiones no deben extrapolarse a pliegos reales
ni otros modelos sin otra evaluación. El script no arranca ni modifica pods.

### Resultados de referencia (2026-11-02, `anthropic/claude-haiku-5-5`)

Corpus sintético anterior, misma máquina y misma clave. Media por consulta:

| Modo | Consultas | Completas | Segundos | Tokens | Checks | Fuentes esperadas |
|---|---|---|---|---|---|---|
| `client.chat()` | 16 (8 preguntas × 2) | 16 | 7,8 | 61 500 | 82/82 | 42/42 |
| `MultiDocumentSearch`, prompts previos a la release | 16 | 15 | 39,7 | 89 500 | 62/82 | 28/42 |
| `MultiDocumentSearch`, prompts de la release | 4 (preguntas amplias y profundas) | 4 | 33,7 | 79 300 | 32/32 | 16/16 |

Lectura: en este corpus la búsqueda habitual fue más rápida y barata con la misma
calidad; los árboles de diez contratos caben en su contexto. Los 20 checks y 10
fuentes perdidos por la versión previa salen de una sola consulta cuya síntesis se
cortó por longitud (`max_output_tokens` era 2 048; ahora 4 096 por defecto) y de
citas anotadas como `[E000008, pág. 12]`, que el evaluador no reconocía (ahora
`cited_ids`). El aviso `children` redujo las páginas leídas en las preguntas
amplias de 300 a 62–70 sin perder respuestas ni fuentes. El modo conjunto sigue
costando varias llamadas por consulta: su valor está en el recorrido explícito y
los límites por llamada cuando las estructuras no caben, no en la velocidad.
