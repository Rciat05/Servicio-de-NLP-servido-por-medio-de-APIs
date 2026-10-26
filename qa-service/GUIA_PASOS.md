# Guía paso a paso (nivel "desde cero")

Convención: los comandos son para **PowerShell** en Windows (tu caso). Lo que va entre `<...>` lo reemplazas tú.
Todos los comandos se ejecutan desde la **carpeta raíz del repo** (la que contiene `README.md`), salvo que diga otra cosa.

---

## Fase 0 — Subir el esqueleto al repo

1. Crea el repo en GitHub y clónalo (esto ya lo sabes).
2. Descomprime el contenido de `qa-service.zip` **dentro** de la carpeta del repo clonado. Comprueba que `.gitignore` queda en la raíz del repo, no dentro de una subcarpeta `qa-service/`.
3. **Qué es `.gitignore` y por qué importa.** Es un archivo de texto donde cada línea es un patrón de archivos que Git debe *fingir que no existen*: nunca aparecen en `git status`, nunca se suben con `git add .`. Las líneas más importantes del nuestro:
   - `.env` → el archivo donde guardas el token. **Si se sube una vez, queda en el historial para siempre** (y la guía dice que se revisa automáticamente). Por eso es la primera línea.
   - `models/*` → los pesos del modelo (cientos de MB). La guía pide excluirlos del repo.
   - `.venv/` → tu entorno virtual (miles de archivos de librerías, no son tuyos).
   - `__pycache__/` → basura que genera Python.
   - `logs/` **no** está ignorado a propósito: los logs y curvas de pérdida son entregables.
4. **Comprueba que funciona** (hazlo, es la forma de entenderlo):
   ```powershell
   Copy-Item .env.example .env
   git status
   ```
   `.env` **no** debe aparecer en la lista. Si quieres ver qué regla lo ignora: `git check-ignore -v .env`.
5. Primer commit:
   ```powershell
   git add .
   git status        # lee la lista antes de seguir: no debe haber .env, .venv ni models/*
   git commit -m "Add project skeleton"
   git push
   ```
6. Reglas de equipo para no perder puntos: commits pequeños y frecuentes, cada quien con su usuario de Git; `git pull` antes de `git push`; **nunca** `--force` sobre `main` ni squash. Si el repo es privado, invita al instructor como colaborador.

## Fase 1 — Entorno de Python (una sola vez)

Usa **Python 3.12** (3.14 es muy nuevo y las librerías de IA pueden no tener paquetes listos). Instálalo desde python.org marcando "Add to PATH" y verifica: `py -3.12 --version`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Si PowerShell se queja de "scripts deshabilitados": `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y repite. Sabrás que está activo porque la línea empieza con `(.venv)`. **Cada terminal nueva requiere activarlo de nuevo.**

```powershell
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt
pytest
```
Debe terminar con `29 passed`. Eso confirma que el código base (API, contrato, post-proceso, tokenización) funciona en tu máquina.

## Fase 2 — Un modelo de juguete para desarrollar la API

Todavía no hay modelo entrenado, así que entrena uno diminuto en tu CPU. **Su calidad será mala; solo sirve para probar la tubería.**

```powershell
python train/train.py --model distilbert-base-uncased --max_train_samples 1000 --val_samples 200 --epochs 1 --run_name smoke
```
(Tardará unos minutos; en tu CPU quizá 10–20. Descarga SQuAD y DistilBERT la primera vez.) Resultado: `models/smoke/` y `logs/smoke/`. Ambos están ignorados / son de prueba: no hace falta commitear `models/smoke`.

## Fase 3 — Levantar la API y probarla

Edita `.env` (el que creaste en la Fase 0) con el **token de desarrollo** que te dé el instructor (no venía en el zip), tu nombre de equipo y `MODEL_DIR=models/smoke`. Luego:

```powershell
python scripts/run_server.py
```
Espera a ver `model loaded, service ready`. En una **segunda terminal** (activa el venv otra vez):
```powershell
$env:QA_API_TOKEN="<token de desarrollo>"
python scripts/check_service.py --url http://127.0.0.1:8000
```
Verás los 5 tests del PDF + el sample público + latencia por petición y la proyección a 50 ítems contra el tope de 150 s. Con el modelo de juguete las respuestas serán malas; lo que importa es que todo diga `PASS`.

## Fase 4 — Túnel público

**Principal: ngrok** (dominio fijo gratis → no tienes que reenviar la URL).
1. Cuenta gratis en ngrok.com, instala (`winget install ngrok.ngrok`) y registra tu authtoken: `ngrok config add-authtoken <tu authtoken>`.
2. En el dashboard de ngrok reclama tu dominio estático gratuito (`<algo>.ngrok-free.app`).
3. Con la API corriendo, en otra terminal: `ngrok http --url=<algo>.ngrok-free.app 8000` (si tu versión es antigua, la bandera es `--domain`).

**Respaldo: Cloudflare** (la guía pide probar uno de respaldo): `winget install Cloudflare.cloudflared` y `cloudflared tunnel --url http://localhost:8000`. La URL cambia cada vez.

**Prueba desde otra red** (obligatorio): conecta *otra* laptop al hotspot de tu celular, o usa tu celular con datos móviles, y corre
`python scripts/check_service.py --url https://<tu-url-pública>`. El dry run debe hacerse por el túnel, no por localhost.

**Laptop de servicio:** enchufada, sin suspensión (`powercfg /change standby-timeout-ac 0`), plan de energía en "Alto rendimiento". La guía permite usar la laptop de cualquier integrante: **sirve con la que mejor latencia dé en `check_service.py`**.

## Fase 5 — Entrenamiento real (tus compañeros, en GPU)

Cada compañero entrena **un** modelo. En Kaggle (Notebook → Accelerator: GPU T4/P100, Internet: On) o Colab (Runtime → T4 GPU):

```bash
!git clone https://github.com/<usuario>/<repo>.git
%cd <repo>
!pip install -r requirements-train.txt
```
(Repo privado: usen un token de solo lectura y **no lo dejen escrito en un notebook compartido**.)

**Antes del entrenamiento completo, prueba con el 10%** (~13 000 preguntas, unos minutos). Aquí tu idea del 10% sí tiene sentido:
```bash
!python train/train.py --model roberta-base --max_train_samples 13000 --epochs 1 --run_name roberta_10pct
```
Si la pérdida baja y no hay errores, lanzan el entrenamiento completo:
```bash
# Compañero A
!python train/train.py --model distilbert-base-uncased --epochs 3 --lr 3e-5 --run_name distilbert
# Compañero B
!python train/train.py --model roberta-base --epochs 2 --lr 3e-5 --run_name roberta
```
Problemas típicos: *CUDA out of memory* → `--batch_size 16 --grad_accum 2`; *loss = nan* → añade `--no_amp`.

Después, para el modelo entrenado (cambia `<run>`):
```bash
!python train/evaluate.py --model_dir models/<run> --n 3000
!python train/tune_threshold.py --analysis logs/<run>/dev_analysis.jsonl --write_config models/<run>/qa_config.json
!python scripts/plot_loss.py logs/<run>
```
- `evaluate` guarda los puntajes crudos del modelo; `tune_threshold` elige el umbral de "sin respuesta" **reponderando a 25% de preguntas sin respuesta** (el dev de SQuAD tiene ~50%, el test oculto ~25%) y lo reporta en una mitad *no usada* para ajustar.
- Descarguen `models/<run>/` (comprimido) y **commiteen ellos mismos** `logs/<run>/` (csv, loss.png, threshold_sweep.csv). Los pesos NO van a git: súbanlos a un Google Drive / GitHub Release.

## Fase 6 — Elegir el modelo final

1. Pon la carpeta del modelo ganador en `models/final/` en la laptop de servicio (debe contener `qa_config.json`, `model.safetensors` y los archivos del tokenizer) y en `.env` pon `MODEL_DIR=models/final`.
2. Corre `check_service.py` ahí. Regla: el mejor F1 en el hold-out **cuyo peor caso de latencia** (prueba también un contexto de ~5000 caracteres) quede holgado respecto a 5 s y a ~3 s de promedio. Si RoBERTa va justo, usa DistilBERT; si va bien, RoBERTa.
3. Comparen también ambos en un mini-set propio de 30–50 preguntas sobre las lecturas del curso (mismo formato que `data/public_sample.json`; ejecútenlo con `evaluate.py --source file --file data/<su_archivo>.json`). El test oculto probablemente se parece más a eso que a Wikipedia.

## Fase 7 — Entrega

1. Edita `REPORT.md`: **borra el bloque de instrucciones** (el comentario `<!-- ... -->`), rellena TODAS las secciones, en inglés, ≤ ~1000 palabras. Una sección vacía = reporte no entregado = nota × 0.5. Anota los problemas que vayan surgiendo **con el hash del commit** que los arregló (`git log --oneline`).
   - El hash del commit `submission` no puede escribirse dentro del propio commit; pon el hash del commit anterior o indica "ver tag `submission`".
2. Verificaciones antes de etiquetar:
   ```powershell
   git ls-files | Select-String "models/"      # solo debe salir models/.gitkeep
   git log --all -S"<tu token de desarrollo>" --oneline    # no debe imprimir nada
   ```
3. Etiqueta y sube:
   ```powershell
   git tag submission
   git push origin submission
   ```
   Si tienes que mover la etiqueta después de un cambio: `git tag -f submission` y `git push -f origin submission` (el `-f` solo sobre la etiqueta).
4. Envía al instructor la URL del repo y la URL del túnel.

## Fase 8 — Día de la presentación (checklist)

- [ ] El instructor te da un **token nuevo**: detén el servidor (`Ctrl+C`), cambia `QA_API_TOKEN` (en la terminal con `$env:QA_API_TOKEN="..."`; si estaba en `.env`, edítalo) y **reinicia** el servidor.
- [ ] Espera `service ready`, abre el túnel y manda la URL (si cambió).
- [ ] Corre `check_service.py` contra la URL pública unos 10 minutos antes.
- [ ] Laptop enchufada, sin suspensión, apps pesadas cerradas, **no la toques** durante la evaluación.

## Preguntas probables del Q&A (y dónde está la respuesta en el código)

| Pregunta | Dónde mirar |
|---|---|
| ¿Cómo decide la API devolver `null`? | `src/qa/postprocess.py` → `decide()` y `analyze_windows()` |
| ¿Cómo manejas contextos más largos que el modelo? | `src/qa/features.py` → `encode_windows()` (ventanas con `doc_stride`) y `analyze_windows()` (mejor span global, `null` mínimo) |
| ¿Cómo garantizas que `context[start:start+len(answer)] == answer`? | offsets del tokenizer rápido + `strip` ajustado en `analyze_windows()` + red de seguridad en `server._run()` |
| ¿Dónde se valida el token y de dónde sale? | `src/qa/server.py` → `_check_auth()`, variable `QA_API_TOKEN` |
| ¿Por qué comparas con `hmac.compare_digest`? | Comparación en tiempo constante (evita filtrar el token por tiempos) |
| ¿Cuándo se carga el modelo y qué pasa antes? | `server.py` → `lifespan` (hilo de arranque); `/health` da 503 mientras carga |
| ¿Cómo elegiste el umbral? | `train/tune_threshold.py` (reponderación a 25%, mitad de ajuste / mitad hold-out) |
| ¿Cómo se etiquetan las preguntas sin respuesta al entrenar? | `src/qa/features.py` → `make_train_features()` (posición del `[CLS]`) |
| ¿Por qué no se pad-ea a 384 al servir? | `encode_windows(... padding=True)`: solo hasta la ventana más larga, así los contextos cortos son baratos en CPU |
