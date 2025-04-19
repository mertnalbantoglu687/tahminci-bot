import discord
from discord.ext import commands
import os  
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

IMAGE_DIR = "Resimler"

def get_class(image_path, model_path, labels_path):
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return "Hata: Model veya etiket dosyası bulunamadı.", 0.0

    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

@bot.command()
async def check(ctx):
    if not ctx.message.attachments:
        await ctx.send("Görsel yüklemedin.")
        return
    
    for attachment in ctx.message.attachments:
        file_name = attachment.filename
        file_path = os.path.join(IMAGE_DIR, file_name)
        
        try:
            await attachment.save(file_path)
            await ctx.send(f"Görsel başarıyla kaydedildi.`{file_name}`")

            model_path = "keras_model.h5"
            labels_path = "labels.txt"
            class_name, confidence = get_class(file_path, model_path, labels_path)
            
            """
            # Hata durumunda mesaj
            if "Hata" in class_name:
                await ctx.send(f"⚠️ {class_name}")
                return
            """

            print(f"Tahmin edilen sınıf: {class_name[2:]} - Güven: %{confidence*100:.2f}")

            messages = {
    "0 Kedi": "Bu bir kedi! Sanılanın aksine, birçok kedi sahipleriyle güçlü sosyal bağlar kurar. Hatta bazı araştırmalar, kedilerin insanlarını tanıyabildiğini, onların /n/n"
            "sesini diğer insanlardan ayırt edebildiğini ve bazen üzgün olduklarında teselli etmeye çalıştıklarını gösteriyor. "
            "Kediler, sessiz ve dikkatli avcılardır ve bazen bir av bulduklarında oldukça sabırlı olabilirler. Onlar için biraz alan ve zaman sağlamak önemlidir.",
    "1 Papagan": "Bu bir papağan! Sanılanın aksine, birçok papağan sahipleriyle güçlü sosyal bağlar kurar. Hatta bazı araştırmalar, papağanların insanlarını tanıyabildiğini, onların /n/n"
                    "sesini diğer insanlardan ayırt edebildiğini ve bazen yalnız kaldıklarında sahiplerini çağırmak için seslenebildiklerini gösteriyor. /n/n"
                    "Papağanlar, zeki ve meraklı canlılardır ve genellikle çevrelerindeki olayları dikkatle izlerler. Bazı türler kelime ezberleyip taklit edebilirken, aslında söylediklerini bağlamla /n/n"
                    "eşleştirme becerisine de sahip olabilirler. Onlar için zihinsel uyarı ve ilgi görmek çok önemlidir.",
    "2 Kelebek": "Bu bir kelebek! Sanılanın aksine, kelebekler yalnızca güzel görünümleriyle değil, aynı zamanda doğadaki rolleriyle de çok kıymetlidir. Bazı çalışmalar, kelebeklerin belirli çiçekleri tanıyabildiğini, kokulara /n/n"
                    "ve renklere tepki verebildiklerini ve göç eden türlerin şaşırtıcı yön bulma yeteneklerine sahip olduğunu ortaya koyuyor. Kelebekler, narin ama dirençli canlılardır ve doğada tozlaşma gibi önemli görevler /n/n"
                    "üstlenirler. Onlar için temiz bir çevre, bol çiçekli alanlar ve kimyasal içermeyen doğal ortamlar sağlamak hayatta kalmaları için hayati önem taşır."}

            
            special_message = messages.get(class_name, "Bu sınıf için özel bir mesaj yok.")
            
            await ctx.send(f"🔍 Tahmin: `{class_name[2:]}` (%{confidence*100:.2f}) {special_message}")
        except Exception as e:
            await ctx.send(f"Bir Hata oldu. {str(e)}")

bot.run("")