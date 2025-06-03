"""Setup utilities and helper functions for the ChatGPT Telegram bot."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from telegram import Update, constants, BotCommandScopeAllGroupChats
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    InlineQueryHandler,
    CallbackQueryHandler,
    Application,
    ContextTypes,
    CallbackContext,
)

from utils import get_thread_id, is_allowed, is_within_budget, edit_message_with_retry, error_handler
from openai_helper import localized_text

async def check_allowed_and_within_budget(bot, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                          is_inline=False) -> bool:
    """
    Checks if the user is allowed to use the bot and if they are within their budget
    :param update: Telegram update object
    :param context: Telegram context object
    :param is_inline: Boolean flag for inline queries
    :return: Boolean indicating if the user is allowed to use the bot
    """
    name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
    user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id

    if not await is_allowed(bot.config, update, context, is_inline=is_inline):
        logging.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
        await bot.send_disallowed_message(update, context, is_inline)
        return False
    if not is_within_budget(bot.config, bot.usage, update, is_inline=is_inline):
        logging.warning(f'User {name} (id: {user_id}) reached their usage limit')
        await bot.send_budget_reached_message(update, context, is_inline)
        return False

    return True

async def send_disallowed_message(bot, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
    """
    Sends the disallowed message to the user.
    """
    if not is_inline:
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=bot.disallowed_message,
            disable_web_page_preview=True
        )
    else:
        result_id = str(uuid4())
        await bot.send_inline_query_result(update, result_id, message_content=bot.disallowed_message)

async def send_budget_reached_message(bot, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
    """
    Sends the budget reached message to the user.
    """
    if not is_inline:
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=bot.budget_limit_message
        )
    else:
        result_id = str(uuid4())
        await bot.send_inline_query_result(update, result_id, message_content=bot.budget_limit_message)

async def post_init(bot, application: Application) -> None:
    """
    Post initialization hook for the bot.
    """
    await application.bot.set_my_commands(bot.group_commands, scope=BotCommandScopeAllGroupChats())
    await application.bot.set_my_commands(bot.commands)

def run(bot):
    """
    Runs the bot indefinitely until the user presses Ctrl+C
    """
    application = ApplicationBuilder() \
        .token(bot.config['token']) \
        .proxy_url(bot.config['proxy']) \
        .get_updates_proxy_url(bot.config['proxy']) \
        .post_init(bot.post_init) \
        .concurrent_updates(True) \
        .build()

    application.add_handler(CommandHandler('reset', bot.reset))
    application.add_handler(CommandHandler('help', bot.help))
    application.add_handler(CommandHandler('image', bot.image))
    application.add_handler(CommandHandler('tts', bot.tts))
    application.add_handler(CommandHandler('start', bot.help))
    application.add_handler(CommandHandler('stats', bot.stats))
    application.add_handler(CommandHandler('resend', bot.resend))
    application.add_handler(CommandHandler(
        'chat', bot.prompt, filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)
    )
    application.add_handler(MessageHandler(
        filters.PHOTO | filters.Document.IMAGE,
        bot.vision))
    application.add_handler(MessageHandler(
        filters.AUDIO | filters.VOICE | filters.Document.AUDIO |
        filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO,
        bot.transcribe))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), bot.prompt))
    application.add_handler(InlineQueryHandler(bot.inline_query, chat_types=[
        constants.ChatType.GROUP, constants.ChatType.SUPERGROUP, constants.ChatType.PRIVATE
    ]))
    application.add_handler(CallbackQueryHandler(bot.handle_callback_inline_query))

    application.add_error_handler(error_handler)

    application.run_polling()
