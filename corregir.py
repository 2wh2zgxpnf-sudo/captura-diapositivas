"""Limpia las fotos de una clase capturadas con la página de captura de diapositivas.

Para cada foto ubica la pantalla (con las esquinas marcadas en el celular, que vienen en esquinas.json
dentro del ZIP, o buscándola automáticamente si no las hay), la endereza y la recorta; luego agrupa
las fotos que muestran la misma diapositiva, se queda con la mejor de cada grupo y arma un PDF.

Uso:
    python corregir.py "<clase.zip o carpeta con las fotos>" [carpeta de salida]

Requiere: pip install opencv-python-headless numpy pillow
"""
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LADO_DETECCION = 1200      # la pantalla se busca en una copia reducida a este lado mayor
AREA_MIN = 0.012           # la pantalla debe ocupar al menos esta fracción de la foto
CONTRASTE_MIN = 25         # diferencia de color mínima entre la pantalla y lo que la rodea
CELDAS = (12, 8)           # rejilla para comparar dos diapositivas
NCC_IGUAL = 0.6            # correlación mínima para considerar que una celda muestra lo mismo
MISMA_DIAPOSITIVA = 0.6    # fracción de celdas con contenido que deben coincidir
ANCHO_MAX = 1920


def leer(ruta):
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)


def escribir(ruta, img, calidad=92):
    cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, calidad])[1].tofile(str(ruta))


def ordenar_esquinas(p):
    """Devuelve las 4 esquinas como: arriba-izq, arriba-der, abajo-der, abajo-izq."""
    p = p.reshape(4, 2).astype(np.float32)
    s, d = p.sum(1), p[:, 0] - p[:, 1]
    return np.array([p[s.argmin()], p[d.argmax()], p[s.argmax()], p[d.argmin()]], np.float32)


def candidatos(peq, gris):
    """Contornos que podrían ser la pantalla: por bordes y por zonas de color vivo (pantalla encendida)."""
    for bajo, alto in ((30, 90), (60, 160)):
        bordes = cv2.dilate(cv2.Canny(gris, bajo, alto), np.ones((3, 3), np.uint8))
        yield from cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    hsv = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)
    vivo = ((hsv[..., 1] > 80) & (hsv[..., 2] > 90)).astype(np.uint8) * 255
    vivo = cv2.morphologyEx(vivo, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    yield from cv2.findContours(vivo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def hallar_pantalla(img):
    """Busca el cuadrilátero de la pantalla. Devuelve esquinas en fracciones 0-1 de la foto, o None."""
    h, w = img.shape[:2]
    f = LADO_DETECCION / max(h, w)
    peq = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    gris = cv2.GaussianBlur(cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    lab = cv2.cvtColor(peq, cv2.COLOR_BGR2LAB)
    ph, pw = gris.shape
    mejor, mejor_puntaje = None, 0
    for c in candidatos(peq, gris):
        if cv2.contourArea(c) < AREA_MIN * ph * pw:
            continue
        casco = cv2.convexHull(c)
        quad = cv2.approxPolyDP(casco, 0.03 * cv2.arcLength(casco, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        area = cv2.contourArea(quad)
        if area > 0.97 * ph * pw or cv2.contourArea(casco) < 0.9 * area:
            continue
        q = ordenar_esquinas(quad)
        lados = [np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])]
        ancho, altura = (lados[0] + lados[1]) / 2, (lados[2] + lados[3]) / 2
        if not 1.1 < ancho / max(altura, 1) < 2.6:
            continue
        # lados opuestos de largo parecido: descarta triángulos y trapecios exagerados
        if min(lados[0], lados[1]) < 0.6 * max(lados[0], lados[1]) or min(lados[2], lados[3]) < 0.6 * max(lados[2], lados[3]):
            continue
        # la pantalla se distingue de lo que la rodea: comparar el color interior con un anillo más afuera del marco
        dentro = np.zeros((ph, pw), np.uint8)
        cv2.fillConvexPoly(dentro, q.astype(np.int32), 255)
        g = max(3, int(0.06 * altura))
        nucleo = np.ones((2 * g + 1, 2 * g + 1), np.uint8)
        fuera = cv2.dilate(dentro, nucleo)
        anillo = cv2.dilate(fuera, nucleo) - fuera
        interior = cv2.erode(dentro, nucleo)
        if not interior.any() or not anillo.any():
            continue
        m_in, m_out = np.array(cv2.mean(lab, interior)[:3]), np.array(cv2.mean(lab, anillo)[:3])
        distinta = np.linalg.norm(m_in - m_out)
        saturacion = cv2.mean(cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)[..., 1], interior)[0]
        if distinta < CONTRASTE_MIN or m_in[0] < 60:
            continue
        puntaje = area * distinta * (1 + saturacion / 64)
        if puntaje > mejor_puntaje:
            mejor, mejor_puntaje = q / [pw, ph], puntaje
    return mejor


def resolver_esquinas(quads):
    """Si la cámara estuvo fija usa la misma zona para todas las fotos; si no, cada foto usa la suya (o queda completa)."""
    hallados = [q for q in quads if q is not None]
    if not hallados:
        return quads, 'sin pantalla detectada: se dejan las fotos completas'
    pila = np.stack(hallados)
    mediana = np.median(pila, axis=0)
    dispersion = np.median(np.abs(pila - mediana).max(axis=(1, 2)))
    if len(hallados) >= 0.5 * len(quads) and dispersion < 0.01:
        return [mediana] * len(quads), 'cámara fija: misma zona para todas las fotos'
    return quads, 'la cámara se movió: zona calculada foto por foto; las que no la tienen quedan completas'


def enderezar(img, quad):
    h, w = img.shape[:2]
    if quad is None:
        return img
    q = (quad * [w, h]).astype(np.float32)
    ancho = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    altura = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    for prop in (16 / 9, 4 / 3):   # ajustar a la proporción estándar más cercana
        if abs(ancho / altura - prop) / prop < 0.12:
            altura = ancho / prop
    escala = min(1, ANCHO_MAX / ancho)
    W, H = int(round(ancho * escala)), int(round(altura * escala))
    destino = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, destino), (W, H), flags=cv2.INTER_CUBIC)


def mejorar(img):
    """Estira el contraste y enfoca un poco, sin alterar los colores de la diapositiva."""
    lo, hi = np.percentile(img, (1, 99))
    if hi - lo > 40:
        img = np.clip((img.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    suave = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.5, suave, -0.5, 0)


def huella(img):
    """Versión pequeña y suavizada para comparar diapositivas entre sí."""
    g = cv2.cvtColor(cv2.resize(img, (480, 272), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)


def parecido(a, b):
    """Fracción de celdas con contenido que muestran lo mismo en ambas huellas (tolera a alguien tapando parte)."""
    cx, cy = CELDAS
    ch, cw = a.shape[0] // cy, a.shape[1] // cx
    iguales = con_contenido = 0
    for j in range(cy):
        for i in range(cx):
            pa = a[j * ch:(j + 1) * ch, i * cw:(i + 1) * cw]
            pb = b[j * ch:(j + 1) * ch, i * cw:(i + 1) * cw]
            sa, sb = pa.std(), pb.std()
            if max(sa, sb) < 6:        # celda lisa en ambas: no dice nada
                if abs(pa.mean() - pb.mean()) > 40:
                    con_contenido += 1
                continue
            con_contenido += 1
            if min(sa, sb) < 2:
                continue
            ncc = ((pa - pa.mean()) * (pb - pb.mean())).mean() / (sa * sb)
            if ncc > NCC_IGUAL and abs(pa.mean() - pb.mean()) < 40:
                iguales += 1
    return 1.0 if con_contenido == 0 else iguales / con_contenido


def nitidez(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def elegir(grupo, huellas):
    """De un grupo de fotos de la misma diapositiva, la más parecida a la mediana (la menos tapada); si empatan, la más nítida."""
    if len(grupo) < 3:
        return max(grupo, key=lambda k: nitidez_de[k])
    mediana = np.median(np.stack([huellas[k] for k in grupo]), axis=0)
    dif = {k: np.abs(huellas[k] - mediana).mean() for k in grupo}
    minimo = min(dif.values())
    return max((k for k in grupo if dif[k] <= minimo + 1.5), key=lambda k: nitidez_de[k])


nitidez_de = {}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    entrada = Path(sys.argv[1])
    temporal = None
    if entrada.suffix.lower() == '.zip':
        temporal = tempfile.TemporaryDirectory()
        zipfile.ZipFile(entrada).extractall(temporal.name)
        carpeta = Path(temporal.name)
    else:
        carpeta = entrada
    fotos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not fotos:
        sys.exit('No hay fotos en ' + str(entrada))
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else entrada.with_name(entrada.stem + '_limpio')
    salida.mkdir(parents=True, exist_ok=True)

    # esquinas.json: lo que se marcó en el celular (fracciones 0-1 de cada foto); lo demás se busca automáticamente
    archivo_esquinas = carpeta / 'esquinas.json'
    marcadas = json.loads(archivo_esquinas.read_text(encoding='utf-8')) if archivo_esquinas.exists() else {}
    print(f'{len(fotos)} fotos. Buscando la pantalla…')
    quads = [np.array(marcadas[p.name], np.float32) if marcadas.get(p.name) else hallar_pantalla(leer(p)) for p in fotos]
    sin_pantalla = sum(q is None for q in quads)
    if marcadas:
        modo = f'{sum(bool(marcadas.get(p.name)) for p in fotos)} con esquinas marcadas en el celular'
    else:
        quads, modo = resolver_esquinas(quads)
    print(f'  pantalla ubicada en {len(fotos) - sin_pantalla}/{len(fotos)} ({modo})')

    rectas, huellas = [], []
    for k, (p, q) in enumerate(zip(fotos, quads)):
        r = enderezar(leer(p), q)
        rectas.append(r)
        huellas.append(huella(r))
        nitidez_de[k] = nitidez(r)

    grupos = [[0]]
    for k in range(1, len(fotos)):
        # comparar contra las últimas fotos del grupo: basta que coincida con una
        if max(parecido(huellas[k], huellas[j]) for j in grupos[-1][-3:]) >= MISMA_DIAPOSITIVA:
            grupos[-1].append(k)
        else:
            grupos.append([k])

    paginas = []
    for n, grupo in enumerate(grupos, 1):
        k = elegir(grupo, huellas)
        final = mejorar(rectas[k])
        destino = salida / f'diapositiva_{n:03d}.jpg'
        escribir(destino, final)
        paginas.append(Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)))
        print(f'  {destino.name}  <- {fotos[k].name}  (de {len(grupo)} fotos)')

    pdf = salida / (entrada.stem + '.pdf')
    paginas[0].save(pdf, save_all=True, append_images=paginas[1:], resolution=150)
    print(f'{len(fotos)} fotos → {len(grupos)} diapositivas. PDF: {pdf}')
    if temporal:
        temporal.cleanup()


if __name__ == '__main__':
    main()
