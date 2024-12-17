from PIL import Image

def crear_mosaico(fotos, output="figures/PI-PI-comp.png"):
    # Cargar las imágenes
    imagenes = [Image.open(foto) for foto in fotos]

    # Obtener el tamaño máximo de cada fila y columna
    ancho_total = max(img.size[0] for img in imagenes[:2]) + max(img.size[0] for img in imagenes[2:])
    alto_total = max(img.size[1] for img in imagenes[0::2]) + max(img.size[1] for img in imagenes[1::2])

    # Crear un lienzo vacío para el mosaico
    mosaic = Image.new("RGB", (ancho_total, alto_total))

    # Colocar las imágenes en el lienzo
    mosaic.paste(imagenes[0], (0, 0))  # Esquina superior izquierda
    mosaic.paste(imagenes[1], (max(img.size[0] for img in imagenes[:2]), 0))  # Esquina superior derecha
    mosaic.paste(imagenes[2], (0, max(img.size[1] for img in imagenes[0::2])))  # Esquina inferior izquierda
    mosaic.paste(imagenes[3], (max(img.size[0] for img in imagenes[:2]), max(img.size[1] for img in imagenes[0::2])))  # Esquina inferior derecha

    # Guardar o mostrar el mosaico
    mosaic.save(output)
    mosaic.show()

# Lista de imágenes
fotos = ["figures/BCGD_0.png", "figures/POLICY-BCGD_1.png", "figures/BCGD-PE_norm.png", "figures/BCGD-PI_norm.png"]

# Crear mosaico
crear_mosaico(fotos)
