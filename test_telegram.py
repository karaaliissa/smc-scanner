import os
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()
bot = Bot(os.getenv("TELEGRAM_BOT_TOKEN"))
chat = os.getenv("TELEGRAM_CHAT_ID")
bot.send_message(chat_id=chat, text="✅ Telegram wiring OK!")
print("Sent.")
