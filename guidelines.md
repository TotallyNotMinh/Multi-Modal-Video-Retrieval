### HỘI THI THỦ THÁCH TRÍ TUỆ NHÂN TẠO

#### THÀNH PHÓ HÒ CHÍ MINH NĂM 2026

```
1.1. Truy vấn dạng 1: Tìm kiếm chính xác theo văn bản (Textual Known Item
Search — Textual KIS)
```
##### Đây là nhiệm vụ tìm kiếm sự kiện dựa trên mô tả bằng văn bản.

```
s Nội dung truy vấn: Ban giám khảo cung cấp một mô tả bằng ngôn ngữ tự
nhiên về một sự kiện. Các đội dự thi cần định vị chính xác đoạn video chứa
```
##### sự kiện bằng cách chỉ ra một khung hình bất kỳ thuộc đoạn video đó. Ở

```
vòng sơ tuyeÁ:n, nội dung đoạn mô tả được cung cép sẵn và tron ven.
® Ví dụ: Truy vấn "Tim video về một diễn giả mặc áo đỏ phát biểu tại một
cuộc họp báo ngoài trời, phía sau có nhiều cây xanh." — Kết quả nộp:
video_id = video_abc(.mp4), frame_id = 1500.
```
```
1.2. Truy vấn dạng 2: Truy vin dạng Hỏi—Đáp (Q&A)
```
##### Đây là nhiệm vụ tìm kiếm sự kiện và trích xuất thông tin cụ thể từ video.

```
s Nội dung truy vấn: Ban giám khảo cung cấp một mô tả bằng ngôn ngữ tự
nhiên của một sự kiện và một câu hỏi vê thông tin trong sự kiện này. Các
đội dự thi cân tìm ra chính xác khoảnh khăc liên quan và trả lời câu hỏi.
Câu trả lời có thê băng tiêng Việt hoặc tieng Anh.
® Vídụ: Truy vấn "Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu
người lên sân khâu đề nhận giải thưởng lớn nhât?" — Kêt quả nộp: video_id
=video_xyz(.mp4), frame id = 3450, answer = "5" hoặc "Năm".
```
1.3. Truy vấn dạng 3: Truy xuất và căn chỉnh sự kiện video theo thời gian
(Temporal Retrieval and Alignment of Key Events - TRAKE)

Đây là một nhiệm vụ phức hợp đòi hỏi độ chính xác cao trong cả việc truy xuất
video và căn chinh thời gian của các khoảnh khăc quan trọng.

TRAKE nhằm đánh giá khả năng của một hệ thống trong việc hiểu sâu séc nội
dung video một cách toàn diện, từ boi cảnh chung cho đên từng khoảnh khăc chi
tiet. Nhiệm vụ yêu cau hệ thông không chỉ tìm kiểm một video phù hợp từ một
kho dữ liệu lớn mà còn phai xác định chính xác các khoanh khăc ngữ nghĩa
(semantic keyframe) của một chuỗi sự kiện có cầu trúc bên trong video đó. Nhiệm
vụ được chia thành hai giai đoạn:

```
® Giai đoạn 1 - Truy xuất (Retrieval): Từ một thư viện video lớn, tìm ra
một video duy nhất chứa chuỗi sự kiện khớp nhất với truy vấn.
® Giai đoạn 2 - Căn chinh (Alignment): Đối với video đã truy xue”ẫ~t, xác
định chính xác một khung hình (semantic keyframe) duy nhat cho mỗi giai
đoạn của chuỗi sự kiện.
```

Lưu ý: "Khung hinh ngữ nghĩa” (semantic keyframe) trong truy van này là khoảnh
khắc mang ý nghĩa về nội dung, khác với "I-Frame" là khung hình kỹ thuật trong
các thuật toán nén video đã được cung cấp cho các đội thi.
® Ví dụ - hành động "Nhay cao": chuỗi sự kiện gồm 4 khoảnh khắc:
o Event I - Chạy đà (Approach): khoảnh khắc ban chân đầu tiên chạm
đất và bước qua khỏi vạch xuất phát.
o Event2— Giậm nhảy (Take-off): khoảnh khắc đầu tiên ban chân của
chân giậm nhảy rời hoàn toàn khỏi mặt đất.
o Event 3 — Bay qua xa (Clearance): khoảnh khắc phần hông của vận
động viên & vị tri cao nhất so với xa ngang.

##### o Event 4 - Tiếp đất (Landing): khoảnh khắc đầu tiên bất kỳ bộ phận

```
nào của lưng (từ vai đến hông) bắt đầu chạm vào dém.
```

# 2.PHUONG PHÁP ĐÁNH GIÁ VÒNG SƠ TUYẾN II

## Đối với mỗi truy vấn, đội thi được gửi tối đa 100 câu trả lời. Mỗi câu trả lời sẽ

```
được chấm một điểm gọi là Điểm Tương Quan (R-Score) — thang đo độ chính
xác nhận giá trị từ 0 đến 1 (1: hoàn toàn chính xác; 0: không chính xác; giá trị
trung gian, ví dụ 0.7: chính xác một phần). Điểm cuối cùng cho mỗi truy vấn
(Final Score, mục 2.2) không chỉ dựa trên một câu trả lời duy nhất, mà là trung
```
## bình của những câu trả lời tốt nhất ở nhiều vị trí xếp hạng khác nhau.

```
2.1. Điểm Tương Quan (R-Score)
Cách tính R-Score khác nhau tùy theo từng loại truy vén:
```
## 2.1.1. Truy vấn Textual KIS

```
® Định dạng trả lời (1;): <video_id>, <frame_id>
e Điều kiện: câu trả lời được xem là chính xác néu khớp video (vị = GT.) và
frame_id năm trong khoang đáp án đúng (id; € [s, e]).
R —Score(r;) = I(v; = GT, A id; € [s,e])
```
## Trong đó I(-) là hàm chỉ thị, trả về 1 néu điều kiện đúng và 0 néu sai.

```
® Ví dụ: câu hỏi "Tìm cảnh một người đang mở laptop trong kho video." Đáp
án đúng của BTC: video L01_V001, khung hình từ 500 dén 510.
o L01 V001,505 — Đúng! R-Score = 1.
o L01 V001, 600 — Sai! Khung hình không nằm trong khoảng cho
phép. R-Score = 0.
o L02_V003, 505 — Sai! Sai video. R-Score = 0.
2.1.2. Truy vấn Q&A (Visual Question Answering)
® Định dạng trả lời (1;): <video_id>, <frame_id>, <answer>
® Điều kiện: câu trả lời được xem là chính xác nếu khớp video (vị = GT,),
frame_id năm trong khoảng đáp án đúng (id; € [s, e]), và answer khớp với
đáp án về mat ngữ nghĩa (a; = GT,).
R —Score(r;) = I(v; = GT, A iải € |s,e] A ai = GT,)
® Ví dụ: câu hỏi "Trong video quay canh bữa tiệc, người phu nữ mặc váy đỏ
đang cam ly màu gi?" Đáp án đúng của BTC: video L05_V005, khung hình
800 đên 900, answer là "màu xanh".
o IL05 V005, 888, màu xanh — Hoàn hao! R-Score = 1.
o L05 V005, 888, màu trắng — Sai answer. R-Score = 0.
o L06 V007, 888, màu xanh — Sai video. R-Score = 0.
2.1.3. Truy vấn TRAKE (Temporal-alignment)
® Định dạng trả lời (1;): <video_id>, <frame_id:>, ..., <frame_id,>
® Điều kiện tiên quyết: nếu video_id nộp không khớp với đáp án (vị # GT,),
truy van nhận 0 diém ngay lập tức. Nêu đúng video, diém được tính băng
ti 1€ khung hinh khớp với đáp án; N là tông sô khoảnh khăc trong truy vân.
```

```
R — Score(r;) =
```
###### =I=

```
N
```
## Ệ11(z'z1l-Ij- € [s„2]) (nếu vị = GT,)

```
J=
R —Score(r;) = 0 (néuv; # GT,)
```
Với mỗi khoảnh khắc thứ j trong chuỗi sự kién, đáp án quy định một doan khung
hình [s;, ej] tương ứng w'yj khoanh khăc ngữ nghĩa đó — cùng nguyên tăc xác định
đoạn [s, e] như ¢ truy van Textual KIS và Q&A. Lưu ý là đoạn ứng với khoảnh
khắc ngữ nghĩa này thường rất ngan thông thường là dưới 10 frame. Mot khung
hình nộp (id;,;) được coi là khớp nêu năm trong đoạn [s;, ej] này.

```
® Ví dụ: câu hỏi "Tìm 4 khoảnh khắc chính kh1 vận động viên thực hiện cú
nhảy: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy." Đáp án
đúng của BTC: video L10_V010, mỗi khoảnh khắc được xác định bằng
một đoạn khung hình:
0 Khoảnh khắc 1 (giậm nhay): đáp án trong đoạn [95, 105].
o Khoảnh khắc 2 (bay qua xà): đáp án trong đoạn [145, 155].
```
##### o Khoảnh khắc 3 (tiếp đất): đáp án trong đoạn [195, 205].

```
o Khoảnh khắc 4 (đứng dậy): đáp án trong đoạn [245, 255].
e Câu trả lời của đội thi: L10_V010, 101, 156, 203, 251
o Video: đúng L10_VO010.
```
##### o Khoảnh khắc 1: 101 € [95, 105] — Đúng.

##### 0 Khoảnh khắc 2: 156 & [145, 155] — Sai.

##### o Khoảnh khắc 3: 203 € [195, 205] — Đúng.

##### o Khoảnh khắc 4: 251 € [245, 255] — Đúng.

Kết quả: khớp 3 trên 4 khoảnh khắc — R-Score = 3/4 = 0.75.

2.2. Điểm Cubi Cùng (Final Score)

Đieq:m Cu<'›i Cùng được tính dựa trên nhu:ng câu trả lời tốt nhất của đội thi & các
mốắc xếp hạng (top) khác nhau. Với mỗi ngưỡng k € {1, 5, 20, 50, 100}, hé thong
xác định Top-k R-Score (R@k): diém R-Score cao nhất trong k câu trả lời đầu
tiên.

```
R@k = maxisisttR — Score(r;)}
```
##### Điểm Cuối Cùng là trung binh cộng của 5 giá trị R@k nói trên:

```
1
Final Score = 5 2k € 1,5,20,50,100y È@k
```
##### s Ví dụ: đội thi nộp 100 câu trả lời cho một truy vấn. Câu trả lời đầu tiên có

```
R-Score = 0.5; câu trả lời ở vị trí sô 3 có R-Score = 0.8 (cao nhât trong 100
câu); câu trả lời ¢ vị trí sô 15 có R-Score = 0.6; các câu trả lời còn lại thap
hơn.
```
##### o Top 1=0.5 (câu trả lời đầu tiên).


```
o Top 5 = Top 20 = Top 50 = Top 100 = 0.8 (câu số 3 vẫn là cao nhất
trong mọi ngưỡng từ 5 trở lên).
```
Final Score = (0.5 + 0.8 + 0.8 + 0.8 + 0.8) / 5 = 0.74 điểm.

Cách tính fflễm này khuyến khích đội thi không chỉ tìm ra một câu trả lời đúng,
mà còn phải xêp nó ở những vị trí đâu tiên trong danh sách trả lời của mình!


##### Dữ liéu cung cấp cho các đội thi để làm quen với bài toán là một phần dữ liệu từ

```
cuộc thi AIC 2026, gồm các thành phần sau:
® Videos: Chứa video được cung cấp.
® Keyframes: Chứa tất cả keyframe được trích xuất từ video được cung cấp
ở trên. Keyframe được lưu trong thư mục tương ứng với tên ñle video —
ví dụ, các keyframe của video L01_V001.mp4 được lưu trong thư mục
L01_V001. Tên các file keyframe được đặt theo thứ tự tăng dần; vị trí
(frame index) tương ứng của mỗi keyframe được ghi trong file metadata.
```
##### ® Objects: Chứa file JSON liệt kê tất cả vật thé (object) phát hiện được từ

```
mo hình Faster R-CNN pretrained trên OpenImages V4. Chi tiết định dạng
kết quả phát hiện vật thé có thể xem ví du tại hướng dẫn của TensorFlow.
Tên ñle JSON tương ứng với tên file keyframe — ví dụ, keyframe
L01_V001/0000.jpg sẽ có file JSON chứa thông tin object là
L01_V001/0000.json.
® CLIP features: Chứa CLIP features (trich xuất từ mô hinh clip-ViT-B-32)
của tất cả các khung hình trong thư mục Keyframes. Toàn bộ CLIP features
của các keyframe được lưu trong một file .npy duy nhất, với thứ tự các
vector feature tăng dẫn tương ứng với chỉ số của keyframe.
® Metadata: Thông tin metadata của video được lấy từ YouTube của kênh
cung cấp dữ liệu. Metadata của mỗi video là một file JSON có tên tương
ứng với tên file video — ví dụ, video L01_V001.mp4 sẽ có file metadata
là L01 V001.json. Một số video trong dữ liệu cung cấp có thé không có
file metadata tương ứng.
```
```
Download dữ liệu tại link:
https://docs.google.com/spreadsheets/d/1rfn1fieTThS Ki3SIoJ6uXOx2AhMq
wGCak6W41ZyZM/edit?usp=sharing
```
Lưu ý:

```
s Dữ liệu thi chính thức là Video; các thành phần còn lại (Keyframes,
Objects, CLIP features, Metadata) chỉ nhằm mục đích cung cấp thêm thông
tin hoặc hỗ trợ xây dựng giải pháp mẫu cho thí sinh.
s Đây cũng là dữ liệu batch 1 của AIC 2025. Dữ liệu đầy dii của vòng sơ
tuyén AIC 2026 sẽ bao gồm thêm dữ liệu batch 2, dự kién được thông báo
cho các đội thi trong thoi gian tới.
```


