import pandas as pd
import joblib  # Для загрузки модели
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ConversationHandler, MessageHandler, filters

# Загрузка обученной модели из файла
model = joblib.load('titanic_model.pkl')  # Убедитесь, что путь к модели правильный

# Определяем этапы для ConversationHandler
AGE, PCLASS, GENDER, SIBSP, PARCH, FARE, EMBARKED = range(7)


async def start(update: Update, context):
    await update.message.reply_text("Привет! Какой у вас возраст?")
    return AGE


async def age(update: Update, context):
    context.user_data['age'] = int(update.message.text)
    await update.message.reply_text("Какой у вас был класс на Титанике? (1-й, 2-й, 3-й класс)")
    return PCLASS


async def pclass(update: Update, context):
    context.user_data['pclass'] = int(update.message.text)
    await update.message.reply_text("Какой у вас пол? (м/ж)")
    return GENDER


async def gender(update: Update, context):
    context.user_data['gender'] = 1 if update.message.text.lower() == 'м' else 0
    await update.message.reply_text("Сколько у вас было братьев/сестер или супругов на борту?")
    return SIBSP


async def sibsp(update: Update, context):
    context.user_data['sibsp'] = int(update.message.text)
    await update.message.reply_text("Сколько у вас было родителей или детей на борту?")
    return PARCH


async def parch(update: Update, context):
    context.user_data['parch'] = int(update.message.text)
    await update.message.reply_text("Сколько стоил ваш билет?")
    return FARE


async def fare(update: Update, context):
    context.user_data['fare'] = float(update.message.text)
    await update.message.reply_text("В каком порту вы сели на Титаник? (C — Шербур, Q — Квинстаун, S — Саутгемптон)")
    return EMBARKED


async def embarked(update: Update, context):
    port = update.message.text.lower()
    if port == 'c':
        context.user_data['embarked'] = 0
    elif port == 'q':
        context.user_data['embarked'] = 1
    else:
        context.user_data['embarked'] = 2

    # После того как все данные собраны, можно сделать предсказание
    features = pd.DataFrame([[
        context.user_data['pclass'],
        context.user_data['gender'],
        context.user_data['age'],
        context.user_data['sibsp'],
        context.user_data['parch'],
        context.user_data['fare'],
        context.user_data['embarked']
    ]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    prediction = model.predict(features)[0]

    if prediction == 1:
        await update.message.reply_text("Вы бы выжили на Титанике!")
    else:
        await update.message.reply_text("К сожалению, вы бы не выжили на Титанике.")

    return ConversationHandler.END


async def cancel(update: Update, context):
    await update.message.reply_text('Диалог был прерван.')
    return ConversationHandler.END


# Основная функция для запуска бота
if __name__ == '__main__':
    application = ApplicationBuilder().token("7931309885:AAEb5Fj92yAIcbqF845UACjaXS2LoGUcfRM").build()

    # Определяем обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, age)],
            PCLASS: [MessageHandler(filters.TEXT & ~filters.COMMAND, pclass)],
            GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, gender)],
            SIBSP: [MessageHandler(filters.TEXT & ~filters.COMMAND, sibsp)],
            PARCH: [MessageHandler(filters.TEXT & ~filters.COMMAND, parch)],
            FARE: [MessageHandler(filters.TEXT & ~filters.COMMAND, fare)],
            EMBARKED: [MessageHandler(filters.TEXT & ~filters.COMMAND, embarked)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    application.add_handler(conv_handler)

    print("Бот запущен...")
    application.run_polling()
