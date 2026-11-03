<div align="center">
  
<a href="https://vectify.ai/pageindex" target="_blank">
<img width="1471" alt="pi_github_banner_low" src="https://github.com/user-attachments/assets/bae02956-6c4e-4a0b-adea-257b0be4aaa1" />
</a>

<br/>
<br/>

<p align="center">
  <a href="https://trendshift.io/repositories/14736" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14736" alt="VectifyAI%2FPageIndex | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# PageIndex: Vectorless, Reasoning-based RAG

<p align="center"><b>Reasoning-based RAG&nbsp; ◦ &nbsp;No Vector DB, No Chunking&nbsp; ◦ &nbsp;Context-Aware Retrieval&nbsp; ◦ &nbsp;Reads Like a Human</b></p>

<h4 align="center">
  <a href="https://pageindex.ai/developer">🌐 Website</a>&nbsp; • &nbsp;
  <a href="https://developer.pageindex.ai/">☁️ Cloud</a>&nbsp; • &nbsp;
  <a href="https://docs.pageindex.ai">📖 Docs</a>&nbsp; • &nbsp;
  <a href="https://pageindex.ai/blog">📝 Blog</a>&nbsp; • &nbsp;
  <a href="https://pageindex.ai/contact">✉️ Contact</a>&nbsp;
</h4>
  
</div>



## Fork tendios/PageIndex

Este fork parte de [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) `v0.5.1` y se publica con etiquetas `v<upstream>-tendios.<n>` (versión del paquete `<upstream>+tendios.<n>`); fija una etiqueta, nunca `main`:

```bash
pip install "pageindex[sql,s3] @ git+https://github.com/tendios/PageIndex@v0.5.1-tendios.3"
```

### Diferencias con upstream

Las extensiones de indexado se activan en modo local; el modo cloud conserva su API.

- **Almacenes** (`pageindex/stores/`): `PageIndexClient(index={"store": ...})` acepta `FileDocumentStore` (el de siempre, por defecto), `S3DocumentStore` (extra `[s3]`, compatible con MinIO) o `SqlDocumentStore` (extra `[sql]`), una instancia o `{"type": "file"|"s3"|"sql", ...}`. Mismo JSON y mismos ids en todos; `pageindex.stores.copy(origen, destino)` migra un índice. Ver [Storage backends](#storage-backends).
- **`prefix` / `schema` en `SqlDocumentStore`**: `prefix="lib_a_"` antepone el prefijo a las tres tablas (documentos, `pageindex_library`, `pageindex_store_lock`) y `schema="lib_a"` (Postgres) las pone en ese esquema; con el mismo URL, cada prefijo o esquema es un almacén independiente con su propio cerrojo.
- **`doc_id` externo**: `submit_document(..., doc_id="contrato-42")` indexa con ese id (1–128 caracteres `[A-Za-z0-9._-]`). Si ya existe, error 409.
- **`if_exists="replace"`**: con `doc_id`, reindexa y sustituye el documento bajo el mismo id.
- **Vista de biblioteca**: con `doc_id` como lista, el primer mensaje del chat lleva, por documento, nombre, metadatos, descripción y árbol (sin texto), recortado a niveles 1-2 por encima de `client.library_view_max_tokens` (12 000). La herramienta `get_library_structure(doc_names)` devuelve lo mismo en una llamada. `client.library_view = False` vuelve al comportamiento de upstream. Opcional: `client.build_library_topics()` construye un índice de temas con el modelo de indexado y `client.library_topics = True` lo muestra.
- **LiteParse opcional**: OCR por página con `pdf-inspector`, contenido compartido por el índice y el chat, bloques y cajas de palabras persistidos. Ver la configuración siguiente.
- **Modo `headings`**, el modo por defecto con LiteParse: el árbol sale de los encabezados del propio documento (los bloques de LiteParse, su numeración, sus páginas de índice y los títulos de la maquetación) y Jev, de Typesafe, decide las líneas dudosas cuando hay `TYPESAFE_AI_API_KEY`. Sin optimizador. Ver [Modo `headings`](#modo-headings-el-árbol-desde-los-encabezados-del-documento).

### LiteParse, OCR y citas con coordenadas

Instalación de la versión del fork con LiteParse:

```bash
pip install "pageindex[sql,s3,liteparse] @ git+https://github.com/tendios/PageIndex@v0.5.1-tendios.3"
```

Para desarrollar desde este checkout:

```bash
pip install -e '.[liteparse]'
```

```python
from pageindex import PageIndexClient

# ANTHROPIC_API_KEY en el entorno; también admite otros modelos/proveedores.
client = PageIndexClient(
    index={
        "model": "anthropic/claude-haiku-5-5",
        "pdf_parser": {"name": "liteparse", "ocr_mode": "auto", "ocr_language": "spa"},
        "storage_path": ".pageindex",
    },
    chat="anthropic/claude-haiku-5-5",
)
doc_id = client.submit_document("contrato.pdf")["doc_id"]
response = client.chat("¿Dónde se entrega el suministro? Cita el documento.",
                       doc_id=doc_id, protocol="chat_completions", citations=True)
answer = response["choices"][0]["message"]["content"]
citations = client.get_citations(answer, doc_id=doc_id)
# Las citas válidas por bloque incluyen page, block_id, bbox y text.
locations = client.get_text_locations(doc_id, "Almacén General", pages="18")
# Cada coincidencia literal devuelve cajas por palabra/glifo y página física.
```

`pdf_parser="liteparse"` es el atajo de esa configuración. También existe el argumento plano `PageIndexClient(model="...", pdf_parser="liteparse")`. Omitirlo, o usar `"default"`, conserva PyPDF2/PDFium. No se reescriben los documentos ya indexados: necesitan reindexación para añadir OCR y geometría.

- `ocr_mode`: `"auto"` inspecciona y repite solo las páginas que necesitan OCR; `"enabled"` habilita el OCR de LiteParse en todas las páginas; `"disabled"` usa solo texto nativo. Por defecto usa Tesseract local; `ocr_server_url` permite el servidor HTTP compatible usado en `rag`. El idioma predeterminado es `spa`.
- `text_format`: `"auto"` usa bloques/Markdown y conserva el texto espacial cuando detecta varias columnas que LiteParse no ha reconocido como tabla. `"markdown"` y `"spatial"` permiten fijar la representación. Conservar columnas no garantiza interpretar correctamente una tabla.
- Flash mantiene PDFium para detectar la jerarquía y recibe el texto de LiteParse para expandir y resumir. Si no hay jerarquía y hay más de diez páginas, el cliente recurre al indexador standard con ese mismo texto OCR. Si has fijado opciones de resumen exclusivas de Flash, pide retirarlas en vez de ignorarlas. `mode="standard"` también admite el adaptador directamente.
- `get_ocr(doc_id)["features"]["pdf_extraction"]` expone versiones, tiempos, páginas dirigidas a OCR y, cuando aplica, `structure_backend="standard_fallback"`. Cada página conserva `layout` con los bloques originales, palabras, dimensiones, rotaciones del texto y confianza OCR. No se envían las coordenadas al modelo.
- `get_block()` y `get_citations()` usan el contrato existente: `bbox=[x0,y0,x1,y1]`, origen superior izquierdo y coordenadas **0–1000** relativas a la página mostrada. `bbox_points` y `layout` conservan las coordenadas originales en puntos a 72 DPI; la rotación del PDF ya está aplicada. Los identificadores de bloque son estables dentro del índice guardado, no entre reindexaciones.
- `get_text_locations()` busca texto literal con normalización Unicode y de espacios, devuelve **todas** las coincidencias y no adivina paráfrasis ni citas con elipsis. Los PDF antiguos sin geometría devuelven `[]`. Las cajas suelen ser de palabras; el motor puede devolver glifos o una caja de línea. Una coincidencia sobre OCR valida el texto reconocido, no su exactitud respecto a la imagen.
- Una cita por bloque puede abarcar un párrafo, una tabla o toda una página cuando solo se conserva texto espacial. Para resaltar una frase concreta, usa `get_text_locations()`. Un bloque inexistente o citado en una página incorrecta no recibe bbox local.

El extra es independiente de `rag` y no cambia `pageindex-service`; el servicio puede adoptarlo al instalar esta revisión y pasar `pdf_parser` al cliente. Su validación de entrada también debe admitir PDF escaneados cuando se active OCR. Los bbox aumentan el tamaño del índice almacenado; el tiempo de OCR y el del modelo se miden por separado.

La librería aporta extracción y geometría genéricas. La recuperación de evidencias para reglas, el registro de cobertura y la caché pertenecen a la capa de servicio; las reglas y los veredictos de coherencia pertenecen al consumidor. Estas funciones no se activan al indexar o consultar un documento.

### Modo `headings`: el árbol desde los encabezados del documento

Con `pdf_parser="liteparse"`, `submit_document(path)` indexa en modo `headings` salvo que se pida `flash` o
`standard` (con el lector predeterminado el modo por defecto sigue siendo `flash`, como en upstream; `headings` exige
LiteParse). El árbol se construye con los encabezados que LiteParse marca (`#`, `##`…), la numeración del propio
documento («1.2», «Cláusula 3», «Annex 2»), las entradas de sus páginas de índice localizadas en el cuerpo y los
títulos que detecta la maquetación de flash; descarta encabezados y pies repetidos, celdas de tabla e ítems de lista,
y un juez decide las líneas que ninguna regla resuelve. Después siguen los resúmenes y la descripción de flash, sin
optimizador: un apartado corto sigue siendo un apartado, porque así se pide el documento.

```python
client = PageIndexClient(index={"mode": "local", "model": "anthropic/claude-haiku-5-5",
                                "pdf_parser": "liteparse"})  # heading_judge="auto": Jev si hay TYPESAFE_AI_API_KEY
client.submit_document("pcap.pdf")                            # mode="headings" por defecto con LiteParse
```

Cómo se decide cada línea:

- **Candidatos**: los bloques de LiteParse, las filas de tabla numeradas en secuencia (los apartados de un cuadro
  resumen; las tablas de inventario con cantidades no cuentan), las líneas numeradas dentro de un bloque y los títulos
  que el lector pegó a otro texto («…DEL CONTRACTE Dinovena. Condicions…», «…de crèdit. 3.5 S'han…»), sin cortar en
  una cita («art. 22.1»). Las páginas de índice se detectan por sus puntos de relleno o porque sus títulos reaparecen
  después en el mismo orden; sus entradas no son nodos.
- **Columna de numeración**: cada familia de marcadores (decimal «2.3», letra «B.1», romano, ordinal en castellano y
  catalán «Vigésima primera», «Vint-i-setena», palabra clave «Cláusula 12», «Sección primera», «Annex III») se sigue
  como lo haría un lector: una línea continúa la secuencia (siguiente hermano, primer hijo, un hueco corto) o no. Un
  reinicio exige aspecto de título y continuación. Una cláusula fija el número de sus subapartados («Novena» es el padre
  de «9.1»); los «1., 2., 3.» que empiezan de nuevo dentro de una cláusula y se leen como frases son su lista.
- **Estilo**: negrita y tamaño de letra de LiteParse frente a los del cuerpo (la negrita no cuenta si el cuerpo ya lo es),
  marca de encabezado del lector y títulos de la maquetación de flash.
- **Huecos**: si la numeración salta (8.3 → 8.5, o una secuencia que empieza en 3), el número que falta se busca en el
  texto entre sus vecinos y se añade si aparece una sola vez.
- **Señales del PDF**: las etiquetas de título de un PDF etiquetado (H1-H6, como exporta Word sus estilos), solo si el
  documento las pone en títulos y no en texto; y los títulos de pdf-inspector como segunda opinión sobre una línea.
- **Juez**: lo que ni la numeración ni el estilo resuelven lo decide Jev, de Typesafe, por líneas o por grupos de
  hermanos numerados. Añade un título a partir de 0,60 de probabilidad y solo quita uno a partir de 0,85; si falla una
  petición, deciden las reglas y el indexado sigue. Después revisa el árbol terminado: cada título sin número, o de
  primer nivel que no forma una secuencia completa, se le pregunta frente al resto del índice y se quita si es
  contenido con 0,85 (`heading_judge={"verify": False}` lo desactiva). `"images": True` le envía además un recorte de
  la página con la línea.
- **Niveles**: por familia (Libro > Título/Parte > Capítulo > Sección > cláusula > subapartados decimales), con los
  contenedores sin número («CUADRO RESUMEN», «QUADRE DE CARACTERÍSTIQUES») sobre lo que abren.

Medido en 14 pliegos reales (664 páginas, castellano y catalán, PCAP, PPT y cuadros resumen; 1.232 títulos de
referencia anotados por Claude Opus 5.5 con tres muestras y adjudicación de los desacuerdos):

| | Encontrados | Precisión | F1 | Padre correcto |
| --- | --- | --- | --- | --- |
| `0.5.0-tendios.9`, con Jev | 66,4 % | 79,5 % | 72,4 % | 59,9 % |
| `0.5.1-tendios.2`, con Jev | 89,0 % | 90,4 % | 89,7 % | 83,1 % |
| esta versión, solo reglas | 91,7 % | 89,0 % | 90,3 % | 80,4 % |
| esta versión, Jev con revisión final | 90,1 % | 91,2 % | 90,6 % | 82,8 % |

Los modos de la librería con modelo, sobre el mismo corpus: `flash` (maquetación y optimizador con Haiku 5.5) 39,8 %
de F1; `standard` (el modelo escribe el índice) entre 57 y 79 % en los seis documentos medidos, con 70-190 s por
documento.

`get_tree()` devuelve en `features.pdf_extraction.headings` el recuento de candidatos, los decididos por la numeración,
por el juez y por las reglas, y `structure_backend: "headings"`.

LiteParse lee como texto espacial una página que convierte en una tabla de pocas filas con el cuerpo en una celda (la
banda de firma electrónica en el margen lo provoca): sus bloques mezclaban las columnas.

### Idioma de los resúmenes

Los resúmenes de cada nodo y la descripción del documento se escriben en el idioma del propio texto (castellano,
catalán…): los prompts lo piden expresamente, porque sin decirlo el modelo responde en inglés. `summary_language="Spanish"`
en la configuración del índice fuerza un idioma concreto.

### Rebasar sobre upstream

```bash
git remote add upstream https://github.com/VectifyAI/PageIndex.git
git fetch upstream --tags
git switch -c rebase/<versión> main
git rebase upstream/main          # o la etiqueta, p. ej. v0.5.1
python -m pytest -q               # tras resolver conflictos
```

Los commits del fork van por tema (almacenes, `doc_id`, vista de biblioteca, `prefix`, versión, CI, este README), así que cada conflicto queda en un tema. Después: pon la versión en `pyproject.toml` (`<upstream>+tendios.1`), lleva la rama a `main` (`git push --force-with-lease origin rebase/<versión>:main`) y etiqueta `v<upstream>-tendios.1`. `publish.yml` no publica desde el fork.

<details open>
<summary><h2>Updates</h2></summary>

- [Oct '26] 🔥 [**Page Search API**](https://search.pageindex.ai): find the pages that answer each question in about a second, and send your LLM only those. *Cut 97% token cost on document question answering*; more accurate and faster than vector search ([benchmark](https://github.com/VectifyAI/page-search-benchmark)).
- [Aug '26] [**PageIndex SDK**](#quickstart): **local** and **cloud** modes — one client to index, retrieve, and chat on your machine (with your own LLM API key) or on [PageIndex Cloud](https://developer.pageindex.ai/).
- [Aug '26] ⚡ [**PageIndex Flash**](https://pageindex.ai/blog/pageindex-flash): fast tree index generation for text-based PDFs, now the default indexing method in PageIndex SDK local mode.
- [Scale PageIndex to Millions of Documents](https://pageindex.ai/blog/pageindex-filesystem): *PageIndex File System* is a file-level tree indexing layer that lets PageIndex reason over an entire corpus, not just a single document.
- [PageIndex App](https://app.pageindex.ai): a human-like document analysis agent for long professional documents.<!-- Also available via [MCP](https://pageindex.ai/developer) or [API](https://pageindex.ai/developer). -->

</details>



# What is PageIndex?

Are you frustrated with vector database retrieval accuracy for long and complex documents? Vector-based RAG retrieves by semantic **similarity**. But **similarity ≠ relevance** — what retrieval actually needs is relevance, and relevance requires **reasoning**. On professional documents that demand contextual understanding, domain expertise, and multi-step reasoning, similarity search misses what is relevant but not similar, and returns what is similar but not relevant.

Inspired by AlphaGo, **[PageIndex](https://vectify.ai/pageindex)** replaces the vector index with a **hierarchical tree index** and lets an LLM **reason** its way through it, the way a human expert turns to and reads the right section of a long report. Retrieval happens in two steps:

1. **Index**: generate a **tree-structure index** for each document
2. **Retrieve**: agentically **search that tree** with LLM reasoning

<div align="center">
  <a href="https://pageindex.ai/blog/pageindex-intro" target="_blank" title="The PageIndex Framework">
    <img src="https://docs.pageindex.ai/images/cookbook/vectorless-rag.png" width="70%">
  </a>
</div>


### TL;DR

<blockquote>PageIndex is a <b>vectorless</b>, <b>reasoning-based RAG</b> engine that <b>mirrors how humans read</b>, delivering <b>traceable</b>, <b>explainable</b>, and <b>context-aware</b> retrieval, with <b>no vector DBs</b> or <b>chunking</b>.</blockquote>

### Compare with Vector RAG

| | Vector RAG | **PageIndex** |
|---|---|---|
| **Index** | vector index | tree index |
| **Retrieval** | semantic similarity search | LLM reasoning over the tree |
| **Result** | opaque, “vibe retrieval” | traceable to explicit references |
| **Context** | query embedding only | full context: conversation history, domain knowledge, etc. |

It is ideal for financial reports, legal documents, regulatory filings, technical manuals, medical literature, academic textbooks, and any other long, complex professional document.




# Quickstart

```bash
pip install -U pageindex
```


```python
import os
from pageindex import PageIndexClient

os.environ["OPENAI_API_KEY"] = "your-openai-key"

client = PageIndexClient(
    index="gpt-6-luna",                 # model to build the tree index
    chat="gpt-6-sol",                   # model to search the tree
)
doc_id = client.submit_document("report.pdf")["doc_id"]

answer = client.chat("What was the 2023 operating margin?", doc_id=doc_id)
print(answer)
```

Local documents get a generated `pi-…` id; pass `doc_id=` to use your own (1–128 letters, digits, `-`, `_`, `.`), e.g. `client.submit_document("report.pdf", doc_id="report-2023")`. A taken id is refused (`PageIndexAPIError`, `status_code` 409) unless you pass `if_exists="replace"`, which reindexes the file and replaces the stored document under the same id.

### Model Recommendations

- **`index=`: a basic model is sufficient.** The tree structure itself is extracted from the document layout without an LLM; the index model only summarizes and refines it, which a basic model does well.
- **`chat=`: use the best model you can afford.** The chat model searches the tree to retrieve information. See [Query cost and accuracy](#query-cost-and-accuracy).

**Have a Claude subscription?** PageIndex can use your subscription to call Claude models through Claude Code: usage is covered by your plan, no API key needed. Install with `pip install -U "pageindex[claude]"`, then prefix model names with `claude-code/`, e.g., `index="claude-code/haiku"`, `chat="claude-code/opus"` (see [example](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/pageindex-claude-subscription.py)).

### Storage backends

Local mode stores each document as three JSON files under `./.pageindex` (change it with `storage_path=`). To keep them in S3 or a database instead, pass a `store` in `index=`:

```python
# S3 or any S3-compatible service (MinIO, R2): pip install -U "pageindex[s3]"
client = PageIndexClient(index={"model": "gpt-6-luna", "store": {
    "type": "s3", "bucket": "my-bucket", "prefix": "pageindex",
    "endpoint_url": None,               # e.g. "http://localhost:9000" for MinIO
}})

# Postgres (or SQLite): pip install -U "pageindex[sql]" psycopg
client = PageIndexClient(index={"model": "gpt-6-luna", "store": {
    "type": "sql", "url": "postgresql+psycopg://user:pass@host/db",
    "prefix": "",                       # e.g. "tenant_a_": one store per prefix
}})
```

In one database, `prefix=` (before all three table names: `pageindex_documents`, `pageindex_library`, `pageindex_store_lock`) or, on Postgres, `schema=` keeps independent stores, each with its own lock.

Credentials come from the usual places (boto3's environment and profiles, the database URL). `store` also takes your own `pageindex.stores.DocumentStore` subclass. Every store keeps the same JSON under the same document ids, so `pageindex.stores.copy(source, target)` moves an index between them (see [example](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/pageindex-storage-backends.py)).

### [Use PageIndex through the SDK client →](https://docs.pageindex.ai/getting-started)

Configure other models, streaming, multi-document search, citations, and more.

### [Integrate PageIndex with your own agent →](https://docs.pageindex.ai/sdk/agents)

Drop PageIndex tools into the OpenAI Agents SDK, the Claude Agent SDK, or any other framework.


# Benchmarks

### Local indexing cost and time

Building a tree locally runs **about $0.001 per page** with `gpt-5.6-luna` as the index model, so a 1,000-page textbook costs a little over a dollar and a few minutes, once, and every later question reuses it. PageIndex is designed not to rely heavily on the model used at index time, so in our experiments a basic model does not hurt quality.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/VectifyAI/PageIndex/main/assets/index-cost-dark.png">
  <img src="https://raw.githubusercontent.com/VectifyAI/PageIndex/main/assets/index-cost-light.png" width="75%" alt="Indexing cost against document length, log-log, for nine PDFs from 9 to 1,098 pages. Points track a $0.0011-per-page reference line; the spread around it is text density, not length.">
</picture>
</div>

Indexing time also scales predictably with document length. In the same local setup, the benchmark documents (9 to 1,098 pages) finished in roughly **13 seconds to 4.5 minutes**.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/VectifyAI/PageIndex/main/assets/index-time-dark.png">
  <img src="https://raw.githubusercontent.com/VectifyAI/PageIndex/main/assets/index-time-light.png" width="75%" alt="Indexing time against document length, log-log, for nine PDFs from 9 to 1,098 pages. The measured indexing times range from about 13 seconds to 4.5 minutes and increase predictably with document length.">
</picture>
</div>



### Query cost and accuracy

[**PageIndex Financial QA Benchmark**](https://github.com/VectifyAI/pageindex-benchmarks/tree/main/financial-qa) compares the accuracy and query cost of 34 configurations of 19 LLMs answering 100 questions over 40 Form 10-K filings from 8 companies, with PageIndex doing the retrieval. Cost is measured per 100-question run and excludes one-time indexing.

<div align="center">
<a href="https://github.com/VectifyAI/pageindex-benchmarks/tree/main/financial-qa">
  <img src="https://raw.githubusercontent.com/VectifyAI/pageindex-benchmarks/main/financial-qa/results/accuracy_vs_cost.png" width="90%" alt="PageIndex Financial QA Benchmark: question answering accuracy versus cost per 100-question run across 19 models on 40 Form 10-K filings from 8 companies.">
</a>
</div>

Full results, data, and reproduction instructions are in the [benchmark repo](https://github.com/VectifyAI/pageindex-benchmarks/tree/main/financial-qa).

### Cost per query vs. native PDF input

The alternative to retrieval is handing the model the whole PDF on every question. That cost grows with the document; PageIndex's does not, because it reads only the nodes its reasoning reaches. On documents where both routes return the same answer, native PDF input costs **2.1× more at 52 pages and 16.6× more at 420** (`gpt-5.6-sol`, prompt caching excluded) — and at 805 pages the document no longer fits in the context window at all.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/VectifyAI/PageIndex/main/assets/query-cost-dark.png">
  <img src="https://raw.githubusercontent.com/VectifyAI/PageIndex/main/assets/query-cost-light.png" width="75%" alt="Cost per query relative to PageIndex retrieval, for five PDFs from 52 to 805 pages. Passing the PDF natively costs 2.1x, 3.4x, 7.8x, and 16.6x more at 52, 85, 198, and 420 pages; at 805 pages it exceeds the model's context window.">
</picture>
</div>

### Leading accuracy on FinanceBench

PageIndex reached a state-of-the-art [**98.7% accuracy**](https://vectify.ai/blog/Mafin2.5) on [FinanceBench](https://arxiv.org/abs/2311.11944) (financial document QA benchmark), vastly outperforming vector-based RAG.

<div align="center">
<a href="https://github.com/VectifyAI/Mafin2.5-FinanceBench">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/VectifyAI/PageIndex/main/assets/financebench-dark.png">
  <img src="https://raw.githubusercontent.com/VectifyAI/PageIndex/main/assets/financebench-light.png" width="70%" alt="FinanceBench accuracy: PageIndex 98.7%, vector RAG 50%.">
</picture>
</a>
</div>

Explore the full FinanceBench [evaluation results](https://github.com/VectifyAI/Mafin2.5-FinanceBench) and the [blog post](https://vectify.ai/blog/Mafin2.5).


# PageIndex Cloud

The open-source version is ideal for text-heavy PDFs and local workflows. With **PageIndex Cloud, document indexing and storage run in the cloud**: PageIndex handles parsing, OCR, image understanding, tree-index construction, and managed storage for you. The chat and retrieval layer remains **compatible with your model**, so you can search the cloud-hosted index using the model provider your application already uses.

Moving indexing and storage from Local to Cloud only requires a [PageIndex API key](https://developer.pageindex.ai/):

```python
import os
from pageindex import PageIndexClient

os.environ["PAGEINDEX_API_KEY"] = "your-pageindex-key"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

client = PageIndexClient(
    index="cloud",                       # build and store the index in PageIndex Cloud
    chat="gpt-6-sol",                    # use your preferred compatible model for chat
)
doc_id = client.submit_document("report.pdf", wait=True)["doc_id"]
print(client.chat("What was the 2023 operating margin?", doc_id=doc_id))
```

| | **Local** | **Cloud** |
|---|---|---|
| Handles | Text-based PDFs | Text-based, scanned, and image-rich documents |
| Indexing | On your machine | Managed by PageIndex |
| Storage | Local directory | Cloud storage |
| Citations | Page-level | Block-level |
| OCR & image understanding | — | ✓ |
| [Metadata](https://docs.pageindex.ai/sdk/documents#metadata) | — | ✓ |
| [Folders](https://docs.pageindex.ai/sdk/documents#folders-cloud) | — | ✓ |
| [MCP server](https://docs.pageindex.ai/mcp) | — | ✓ |

### More About PageIndex Cloud

- [Scale PageIndex to Millions of Documents](https://pageindex.ai/blog/pageindex-filesystem): **PageIndex File System** is a Cloud-only, file-level tree indexing layer that lets PageIndex reason over an entire corpus, not just a single document.

### Ready to Try It?

- Get a [PageIndex API key](https://developer.pageindex.ai/)
- Read the [PageIndex Cloud documentation](https://docs.pageindex.ai/)

For dedicated deployment (VPC or on-premises), [contact us](https://pageindex.ai/contact) or [book a demo](https://calendly.com/pageindex/meet).



---

# ⭐ Support Us

Leave us a star 🌟 if you like our project. Thank you!  

<p>
  <img src="https://github.com/user-attachments/assets/eae4ff38-48ae-4a7c-b19f-eab81201d794" width="80%">
</p>

Please cite this work as:
```
Mingtian Zhang, Yu Tang and PageIndex Team,
"PageIndex: Next-Generation Vectorless, Reasoning-based RAG",
PageIndex Blog, Sep 2025.
```

<details>
<summary>Or use the BibTeX citation.</summary>

```bibtex
@article{zhang2025pageindex,
  author = {Mingtian Zhang and Yu Tang and PageIndex Team},
  title = {PageIndex: Next-Generation Vectorless, Reasoning-based RAG},
  journal = {PageIndex Blog},
  year = {2025},
  month = {September},
  note = {https://pageindex.ai/blog/pageindex-intro},
}
```
</details>




---

© 2026 [PageIndex AI](https://pageindex.ai)

<img width="288" height="80" alt="pageindex-wordmark-animated-288-warm" src="https://github.com/user-attachments/assets/e1c677f2-b590-4ffc-859d-e8cc7d794bdf" />
