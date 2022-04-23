from ArabicOcr import arabicocr

image_path = r'C:\Users\Abdullah\Desktop\1.png'
out_image = 'out.jpg'
results = arabicocr.arabic_ocr(image_path, out_image)
print(results)
words = []
count = 0

for i in range(len(results)):
    word = results[i][1]
    print('str is : ', i, word)
    words.append(word)

if str(results[3][1]) == 'محقن':

    print('yes is match ' , str(results[3][1]))
with open('file.txt', 'w', encoding='utf-8') as myfile:
    myfile.write(str(words))
