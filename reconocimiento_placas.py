import cv2
import pytesseract

# Configura la ruta de tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Actualiza esta ruta según tu instalación


# Cargar la imagen
image_path = 'Cars0.png'
image = cv2.imread(image_path)

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar filtro de borde
edged = cv2.Canny(gray, 30, 200)

# Encontrar contornos en la imagen
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Inicializar variable para la ubicación de la placa
plate_contour = None

# Iterar sobre los contornos para encontrar la placa
for contour in contours:
    epsilon = 0.018 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:  # Asumiendo que la placa es un cuadrilátero
        plate_contour = approx
        break

if plate_contour is None:
    print("No se encontró la placa.")
else:
    # Dibujar el contorno en la imagen original
    cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)

    # Crear una máscara y extraer la placa de la imagen
    mask = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.bitwise_not(mask)

    x, y, w, h = cv2.boundingRect(plate_contour)
    plate = mask[y:y + h, x:x + w]

    # Usar Tesseract para reconocer el texto en la placa
    text = pytesseract.image_to_string(plate, config='--psm 8')
    print("Placa detectada:", text.strip())

    # Mostrar la imagen con el contorno de la placa
    cv2.imshow('Image with Plate Contour', image)
    cv2.imshow('Detected Plate', plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

