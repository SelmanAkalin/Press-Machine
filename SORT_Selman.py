import onnxruntime as ort
import numpy as np
import cv2
import os
from sort import *

# 1. Resmi Yükleme ve Ön İşleme
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image = np.expand_dims(image, axis=0)  # Batch boyutu ekle [1, C, H, W]
    image = image.astype(np.float32)
    return image

# 2. İlk Modeli Çalıştırma
def run_model(model_path, input_data):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_data})
    return output[0]

# 3. İlk Modelin Çıktısını Yorumlama
def interpret_output(output):
    classes = ["after", "before", "ng"]
    predicted_class = classes[np.argmax(output)]
    confidence = np.max(output)
    return predicted_class, confidence

# 4. Contour İçindeki ROI'yi Kırpma
def crop_image_by_contour(image, points):
    # Mask oluştur
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    
    # Mask'ı uygula
    masked_image = cv2.bitwise_and(image, mask)
    
    # ROI'yi bul
    x, y, w, h = cv2.boundingRect(points)
    cropped_image = masked_image[y:y+h, x:x+w]
    
    return cropped_image, (x, y, w, h)

# 5. İkinci Model için Ön İşleme
def preprocess_for_second_model(image):
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image = np.expand_dims(image, axis=0)  # Batch boyutu ekle [1, C, H, W]
    image = image.astype(np.float16)
    return image

# 6. Heatmap'i ROI İçinde Çizme ve ROI'yi Görüntüye Ekleme
def overlay_heatmap_on_roi(image, heatmap, roi_coords, points):
    x, y, w, h = roi_coords
    
    # Heatmap'i ROI boyutuna yeniden boyutlandır
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Heatmap'i renklendir
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Mask oluştur (ROI dışındaki bölgeleri sıfırla)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points - np.array([x, y])], 255)  # Mask'ı ROI'ye göre ayarla

    # Heatmap'i mask ile sınırla
    heatmap_colored = cv2.bitwise_and(heatmap_colored, heatmap_colored, mask=mask)
    
    # Heatmap'i doğrudan ROI içine yerleştir (saydamlık olmadan)
    roi = image[y:y+h, x:x+w]
    roi[mask > 0] = heatmap_colored[mask > 0]  # Sadece maskeli bölgeye heatmap'i yerleştir
    
    # ROI'yi görüntü üzerine çiz
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return image

# 7. İkinci Modelin Çıktısını Yorumlama
def interpret_second_output(output):
    classes = ["ag", "an", "bg", "bn"]
    predicted_class = classes[np.argmax(output)]
    confidence = np.max(output)
    return predicted_class, confidence

# Ana İşlem
if __name__ == "__main__":
    # Resmi yükle ve ön işleme yap
    mot_tracker = Sort()
    folder_path = r"path_here"

    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)

        original_image = cv2.imread(image_path)
        input_data = preprocess_image(image_path)

        # İlk modeli çalıştır.
        model1_path = r".onxx_path_here"  # İlk modelin yolunu buraya yazın
        output1 = run_model(model1_path, input_data)
        predicted_class, confidence = interpret_output(output1)


            
            #roi = overlay_heatmap_on_roi(original_image, heatmap, cv2.boundingRect(points), points)
            # ROI'yi görüntü üzerine çiz
        cv2.polylines(original_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Sonucu göster
        cv2.imshow("Heatmap Overlay with ROI", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()