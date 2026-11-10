# ID_Gestos_OpenCV

Reconocimiento de gestos con las manos mediante el uso de OpenCV.
Se hace uso de clasificadores en cascada (Haar cascades), segmentación de piel y detección de contornos para reconocer gestos estáticos de la mano, **sin utilizar modelos de inteligencia artificial** adicionales.

## Objetivos

- Detectar manos en tiempo real a partir de la imagen de una webcam.
- Segmentar la piel en el espacio de color YCrCb y ajustar la segmentación a las condiciones de luz y al tono de piel del usuario.
- Analizar la geometría del contorno de la mano (envolvente convexa y defectos de convexidad) para estimar el número de dedos extendidos.
- Clasificar gestos estáticos sencillos: puño cerrado (`fist`), mano abierta (`open`), gesto parcial (`partial`, p. ej. victoria o señalar) o desconocido (`unknown`).
- Aplicar **filtros de imagen** sobre el vídeo a partir del número de dedos detectados (blanco y negro, Canny, desenfoque gaussiano) mediante una interacción explícita del usuario (tecla `w`).

## Cómo funciona

El flujo se ejecuta fotograma a fotograma en `src/main.py`:

1. **Captura**: `WebcamController` lee un fotograma de la cámara.
2. **Preprocesado**: el fotograma se convierte a escala de grises y se ecualiza el histograma para mejorar el contraste.
3. **Exclusión de caras**: se detectan rostros con `haarcascade_frontalface_alt.xml` para ignorar la piel de la cara.
4. **Máscara de piel**: se convierte la imagen a YCrCb y se umbraliza con un rango (centro ± tolerancia) para obtener una máscara binaria de piel. Se limpia con operaciones morfológicas y se anulan las zonas de las caras.
5. **Detección con Haar cascades**: se aplican varios clasificadores de `models/` (`rpalm`, `lpalm`, `right`, `left`, `fist`, `Hand.Cascade.1`, `palm`). Las candidatas se filtran por área mínima, por solapamiento con caras y por porcentaje de piel dentro de la caja.
6. **Refinado de la caja**: como las cascadas de palma suelen dejar fuera el pulgar, se busca el contorno de piel más grande en una región ampliada alrededor de la caja (parámetro `box_padding`), de modo que la caja ajustada incluya los dedos que sobresalen.
7. **Supresión de no máximos (NMS)**: se eliminan las detecciones solapadas (IoU) quedándose con la de mayor puntuación.
8. **Análisis de contorno** (`ContourAnalyzer`, en `src/core/contour_analyzer.py`):
   - Se recorta la máscara de piel a la región de la detección (con un margen `roi_padding`) y se limpia con morfología.
   - Se toma el contorno más grande como contorno de la mano.
   - Se calculan la envolvente convexa (*convex hull*) y los defectos de convexidad (los "valles" entre dedos).
   - Se filtran los defectos por profundidad (relativa al tamaño de la mano) y por ángulo máximo (90° por defecto).
   - Se calcula la **solidez** (área del contorno / área de la envolvente) y se clasifica el gesto:
     - Solidez > 0.85 y 0 valles → `fist` (0 dedos).
     - 3 o más valles → `open` (valles + 1 dedos, máximo 5).
     - 1 o 2 valles → `partial` (valles + 1 dedos).
     - En otro caso → `unknown`.
9. **Filtros por gestos** (`FilterManager`, en `src/core/filters.py`):
   - El usuario arma el modo de espera pulsando `w` (banner amarillo `WAITING`).
   - En cada fotograma, si el modo está armado, se consulta `detector.last_contours` y se busca una mano cuyo `finger_count` esté en `SUPPORTED_FINGERS`.
   - Si se encuentra, se activa el filtro correspondiente y el modo se desarma (banner verde `ACTIVE FILTER`).
   - El filtro se aplica al fotograma **después** de la detección y el dibujo de cajas/contornos (el detector necesita la imagen en color) pero **antes** de dibujar la ayuda y los banners, de modo que el texto queda en color sobre el efecto.
   - `r` desactiva el filtro (y sigue reseteando el modelo de piel); `w` vuelve a armar el modo desde cero.
10. **Visualización**: se dibujan la caja de la detección (verde), el contorno (magenta), la envolvente (naranja), los defectos (círculos rojos) y la etiqueta `gesto | dedos | solidez`.

## Funciones de OpenCV utilizadas

| Función / clase | Uso en el proyecto |
|---|---|
| `cv2.VideoCapture` (`read`, `isOpened`, `release`) | Captura de vídeo desde la webcam. |
| `cv2.cvtColor` | Conversión BGR → GRAY (para las cascadas) y BGR → YCrCb (segmentación de piel y calibración). |
| `cv2.equalizeHist` | Ecualización del histograma de la imagen en grises. |
| `cv2.CascadeClassifier` (`empty`) | Carga de los clasificadores Haar de manos y de cara. |
| `CascadeClassifier.detectMultiScale` | Detección de caras. |
| `CascadeClassifier.detectMultiScale3` | Detección de manos devolviendo la puntuación (pesos) de cada detección. |
| `cv2.inRange` | Umbralización del color de piel en YCrCb para obtener la máscara binaria. |
| `cv2.morphologyEx` (`MORPH_OPEN`, `MORPH_CLOSE`) | Eliminación de ruido y cierre de huecos en la máscara. |
| `cv2.countNonZero` | Cálculo del porcentaje de píxeles de piel dentro de una caja. |
| `cv2.findContours` (`RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`) | Obtención de los contornos de piel. |
| `cv2.contourArea` | Área del contorno y de la envolvente (selección del contorno mayor y cálculo de solidez). |
| `cv2.boundingRect` | Caja que se ajusta al contorno real. |
| `cv2.convexHull` | Envolvente convexa (como índices y como puntos). |
| `cv2.convexityDefects` | Defectos de convexidad (valles entre dedos). |
| `cv2.rectangle` | Dibujo de las cajas de detección. |
| `cv2.putText` | Etiquetas y texto de ayuda en pantalla. |
| `cv2.drawContours` | Dibujo del contorno y de la envolvente convexa. |
| `cv2.circle` | Marcado de los puntos más profundos de los defectos. |
| `cv2.namedWindow` | Creación de la ventana principal. |
| `cv2.createTrackbar` / `cv2.setTrackbarPos` | Deslizadores de ajuste de la segmentación de piel. |
| `cv2.setMouseCallback` | Calibración de la piel con clic del ratón. |
| `cv2.cvtColor(BGR → GRAY)` + `cv2.cvtColor(GRAY → BGR)` | Aplica filtro blanco y negro. |
| `cv2.Canny` | Aplica filtro canny para resaltar esquinas. |
| `cv2.GaussianBlur` | Aplica filtro gaussiano para suavizar. |
| `cv2.imshow` / `cv2.destroyWindow` / `cv2.destroyAllWindows` | Mostrar y cerrar ventanas. |
| `cv2.waitKey` | Lectura del teclado. |

## Estructura del proyecto

```
models/                         Clasificadores Haar (manos y cara)
src/main.py                     Bucle principal, ventanas, deslizadores y teclas
src/core/webcam_controller.py   Captura de la webcam
src/core/hand_detector.py       Detección de manos y máscara de piel
src/core/contour_analyzer.py    Análisis de contorno, dedos y gesto
src/core/filter.py              Añade filtros según el número de dedos detectados
src/config.py                   Carga/validación de settings.json y atajos de teclado
src/settings.json               Configuración editable por el usuario (se autogenera)
```

## Requisitos y ejecución

- Python 3 con `opencv-python` y `numpy`:

  ```
  pip install opencv-python numpy
  ```

- Cámara: `WebcamController` usa por defecto el índice `1` (`source=1`). Si solo tienes una cámara, cambia el valor a `0` en `src/core/webcam_controller.py`.
- Ejecución desde la carpeta `src`:

  ```
  cd src
  python main.py
  ```
## Configuración de teclas (`settings.json`)

Las teclas de acción se leen desde `src/settings.json`. **La primera vez que se ejecuta el programa, el archivo se genera automáticamente** con los valores por defecto, de modo que el usuario puede reasignar teclas sin tocar el código Python.

Formato del archivo:

```json
{
"_version": 1,
"_comment": "Edita 'key' con la tecla que quieras. Debe ser un único carácter.",
"quit":            { "key": "q", "help": "quit" },
"toggle_mask":     { "key": "m", "help": "mask" },
"reset_skin":      { "key": "r", "help": "reset" },
"toggle_contours": { "key": "c", "help": "contours" }
}
```

- `key`: tecla asignada. Debe ser un único carácter.
- `help`: texto que aparece en la barra de ayuda del HUD. Se regenera automáticamente a partir de este archivo.
- `_version`: reservado para futuras migraciones de formato.
- `_comment`: nota informativa para el usuario (JSON no admite comentarios nativos).

## Cómo usar el programa

### Ventanas

- **Webcam**: imagen de la cámara con las detecciones, los contornos y los deslizadores.
- **Skin mask**: máscara binaria de piel (se muestra/oculta con la tecla `m`).

### Teclas (por defecto)

| Tecla | Acción |
|---|---|
| `q` | Salir del programa. |
| `m` | Mostrar / ocultar la ventana de la máscara de piel. |
| `c` | Activar / desactivar el dibujo de contornos, envolvente y defectos. |
| `w` | Activar detección de filtros pedidos. |
| `e` | Limpiar filtros actuales. |
| `r` | Restablecer los valores de piel por defecto (y los deslizadores). |

    El texto de ayuda que aparece en la ventana se genera automáticamente a partir de settings.json, por lo que siempre refleja las teclas activas.

### Ratón

- **Clic izquierdo sobre la mano** en la ventana *Webcam*: calibra el color de piel tomando la mediana de un pequeño parche alrededor del punto pulsado. Los deslizadores se actualizan automáticamente.

### Deslizadores (ventana Webcam)

| Deslizador | Rango | Descripción |
|---|---|---|
| `Luminosity` | 0 – 255 | Centro del canal de luminancia (Y) de la piel. |
| `Lum range` | 0 – 128 | Tolerancia de luminancia alrededor del centro. |
| `Chroma tol` | 0 – 60 | Tolerancia de los canales de crominancia (Cr y Cb). |

### Consejos para un buen funcionamiento

1. Usa una **iluminación uniforme** y un fondo que no tenga colores parecidos a la piel.
2. Pulsa `m` para ver la máscara de piel y comprueba que la mano aparece completa (incluido el pulgar) en blanco y que el fondo queda en negro.
3. Haz **clic sobre tu mano** para calibrar el color de piel y, si hace falta, afina con `Lum range` y `Chroma tol`. Aumentar las tolerancias incluye más piel pero también más ruido; reducirlas hace lo contrario.
4. Coloca la mano a una distancia media, con la palma hacia la cámara y los dedos separados.
5. Mantén las caras fuera del encuadre o deja que el detector las excluya; evita que el antebrazo ocupe demasiado espacio, ya que baja la solidez.
6. Si cambia la iluminación, pulsa `r` y vuelve a calibrar.
7. Ajusta `box_padding` (`HandDetector`) y `roi_padding` (`ContourAnalyzer`) si el pulgar queda fuera de la caja o se incluye demasiado antebrazo.
