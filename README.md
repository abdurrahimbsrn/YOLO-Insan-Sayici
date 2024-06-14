YOLOv4 ile Canlı İnsan Sayımı

Bu proje, YOLOv4 modelini kullanarak canlı kameradan insanları tespit edip sayan bir C++ uygulamasıdır.

1. YOLOv4 Modelinin Yüklenmesi
Program, YOLOv4 modelinin yapılandırma ve ağırlık dosyalarını kullanarak modeli yükler. Ayrıca, COCO veri kümesindeki sınıf isimlerini içeren bir dosya da yüklenir.

3. Sınıf İsimlerinin Yüklenmesi
COCO veri kümesindeki sınıf isimleri, programın insanları tanıyabilmesi için bir listeye yüklenir.

4. Canlı Kamera Girişi
Program, bilgisayarın bağlı olduğu kamerayı kullanarak canlı video akışını başlatır. Kamera açıldıktan sonra her kare, nesne tespiti için işlenir.

5. Nesne Tespiti ve Sayımı
Her video karesinde, YOLO modeli kullanılarak nesne tespiti yapılır. Tespit edilen nesneler arasında sadece insanlar (classId 0) sayılır. YOLO modeli, her karede bulunan nesnelerin koordinatlarını ve sınıflarını çıkarır. İnsanlar tespit edildiklerinde, bunların koordinatları ve güven skorları saklanır.

6. Sonuçların Görüntülenmesi
Tespit edilen nesneler ve kişi sayısı, her karede ekranda gösterilir. Bu bilgiler, görsel olarak karede işaretlenir ve sayılar görüntülenir. Program, kullanıcı q tuşuna bastığında sonlandırılır.
