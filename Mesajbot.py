from discord.ext import commands
from dotenv import load_dotenv
from Mantık import *
import discord
import random
import asyncio
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import time
import os
import re

load_dotenv()

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True
bot = commands.Bot(command_prefix="", intents=intents)

IMAGE_DIR = "Resimler"

def get_class(image_path, model_path, labels_path):
    # Model ve etiket dosyalarının varlığını kontrol et
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return "Hata: Model veya etiket dosyası bulunamadı.", 0.0
    
    # Modeli yükle
    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r").readlines()
    
    # Görüntü ön işleme
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Tahmin yap
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

@bot.event
async def on_ready():
    print(f"Mesajbot Açıldı.")

@bot.event
async def on_message(message):
    cleaned_content = message.content.lower()
    if message.author == bot.user:
        return

    if re.search(r"\s*m+\s*e+\s*r+\s*h+\s*a+\s*b+\s*a+\s*$", cleaned_content):
        await message.channel.send("Sana da merhaba.")
        begenme = random.randint(1, 2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r"\s*n+\s*a+\s*s+\s*ı+\s*l+\s*s+\s*ı+\s*n+\s*(\?+\s*)?$", cleaned_content, re.IGNORECASE):
        await message.channel.send("İyiyim, sorduğun için teşekkür ederim.")
        begenme = random.randint(1, 2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r"\s*a+\s*d+\s*ı+\s*n+\s*\s*n+\s*e+\s*(\?+\s*)?$", cleaned_content, re.IGNORECASE):
        await message.channel.send("Benim Adım Mesajbot.")
        begenme = random.randint(1, 2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r"\s*i+\s*s+\s*m+\s*i+\s*n+\s*\s*n+\s*e+\s*(\?+\s*)?$", cleaned_content, re.IGNORECASE):
        await message.channel.send("Benim ismim Mesajbot.")
        begenme = random.randint(1, 2)
        if begenme == 1:
            await message.add_reaction("👍🏻")
        await message.channel.send("Beni Mert NALBANTOĞLU yazılımladı.")
        begenme = random.randint(1,2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r'\s*s+\s*a+\s*n+\s*a+\s*\s*g+\s*u+\s*v+\s*e+\s*n+\s*i+\s*b+\s*i+\s*l+\s*i+\s*r+\s*\s*m+\s*i+\s*y+\s*i+\s*m+\s*$',cleaned_content,re.IGNORECASE):
        await message.channel.send("Evet, bana güvenebilirsin.")
        begenme = random.randint(1,2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r'\s*s+\s*e+\s*n+\s*\s*n+\s*e+\s*s+\s*i+\s*n+\s*$',cleaned_content,re.IGNORECASE):
        await message.channel.send("Ben bir yapay zekayım.")
        begenme = random.randint(1,2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r'\s*n+\s*e+\s*r+\s*e+\s*l+\s*i+\s*s+\s*i+\s*n+\s*$',cleaned_content,re.IGNORECASE):
        await message.channel.send("Ben Discord'a aidim.")
        begenme = random.randint(1,2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r'\s*t+\s*e+\s*ş+\s*e+\s*k+\s*ü+\s*r+\s*+\s*e+\s*d+\s*e+\s*r+\s*i+\s*m+\s*$',cleaned_content):
        await message.channel.send("Rica ederim, her zaman yanındayım. Bir sorun olduğunda sormaktan çekinme.")
        begenme = random.randint(1,2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r'\s*p+\s*a+\s*r+\s*o+\s*l+\s*a+\s*\s*b+\s*e+\s*l+\s*i+\s*r+\s*l+\s*e+\s*$',cleaned_content):
        await message.channel.send(Parola(25))
        begenme = random.randint(1,2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r'\s*s+\s*ı+\s*k+\s*ı+\s*l+\s*d+\s*ı+\s*m+\s*$',cleaned_content):
        await message.channel.send("Belki Desen Makinesi sıkıntını giderebilir: https://hub.kodland.org/en/project/284214")
        begenme = random.randint(1,2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r'\s*o+\s*y+\s*u+\s*n+\s*\s*o+\s*y+\s*n+\s*a+\s*t+\s*$',cleaned_content):
        await message.channel.send(Oyun())
        begenme = random.randint(1,2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r'\s*e+\s*m+\s*o+\s*j+\s*i+\s*\s*a+\s*t+\s*$',cleaned_content):
        await message.channel.send(Surat())
        begenme = random.randint(1,2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r"\s*ö+\s*r+\s*d+\s*e+\s*k+", cleaned_content, re.IGNORECASE):
        image_url = Ördek()
        await message.channel.send(image_url)
        begenme = random.randint(1, 2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r"\s*k+\s*ö+\s*p+\s*e+\s*k+", cleaned_content, re.IGNORECASE):
        image_url = Köpek()
        await message.channel.send(image_url)
        begenme = random.randint(1, 2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

    elif re.search(r"\s*s\s*ö\s*y\s*l\s*e\s*", cleaned_content, re.IGNORECASE):
        
        begenme = random.randint(1, 2)
        if begenme == 1:
            await message.add_reaction("👍🏻")

@bot.command()
async def basla(ctx):
    soru = await ctx.send(f"Yazı mı tura mı?")
    def Cevap(mesaj):
        return mesaj.author == ctx.author and mesaj.channel == ctx.channel
    cevap_mesajı = await bot.wait_for("message", check=Cevap)
    cevap = cevap_mesajı.content.lower().strip()
    secim = random.randint(1, 2)
    if cevap == "yazı" or cevap == "tura":
        await ctx.send(f"Parayı atıyorum...")
        await asyncio.sleep(3)
        await ctx.send(f"Attım.")
        if secim == 1:
            with open("Resimler/Resim1.png", "rb") as f:
                picture = discord.File(f, filename="Resim1.png")
            await ctx.send(file=picture)
            if cevap == "yazı":
                await ctx.send(f"Yanlış bildin.")
            elif cevap == "tura":
                await ctx.send(f"Doğru bildin.")
        elif secim == 2:
            with open("Resimler/Resim2.png", "rb") as f:
                picture = discord.File(f, filename="Resim2.png")
            await ctx.send(file=picture)
            if cevap == "yazı":
                await ctx.send(f"Doğru bildin.")
            elif cevap == "tura":
                await ctx.send(f"Yanlış bildin.")
    else:
        await ctx.send(f"Geçersiz seçenek, lütfen 'yazı' veya 'tura' yazın.")

@bot.command()
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_path = os.path.join(IMAGE_DIR, file_name)
            try:
                await attachment.save(file_path)
                await ctx.send(f"Görsel başarıyla kaydedildi: `{file_path}`")
            except Exception as e:
                await ctx.send(f"Görsel kaydedilirken bir hata oldu. {str(e)}")
    else:
        await ctx.send("Görsel yüklemedin.")

bot.run("")
