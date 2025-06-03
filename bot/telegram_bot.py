from __future__ import annotations

import asyncio
import logging
import os
import io

from uuid import uuid4
from telegram import BotCommandScopeAllGroupChats, Update, constants
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle
from telegram import InputTextMessageContent, BotCommand
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, CallbackQueryHandler, Application, ContextTypes, CallbackContext

from pydub import AudioSegment
from PIL import Image

from utils import is_group_chat, get_thread_id, message_text, wrap_with_indicator, split_into_chunks, \
    edit_message_with_retry, get_stream_cutoff_values, is_allowed, get_remaining_budget, is_admin, is_within_budget, \
    get_reply_to_message_id, add_chat_request_to_usage_tracker, error_handler, is_direct_result, handle_direct_result, \
    cleanup_intermediate_files
from openai_helper import OpenAIHelper, localized_text
from usage_tracker import UsageTracker

from .command_handlers import (
    help as help_command,
    stats as stats_command,
    resend as resend_command,
    reset as reset_command,
    image as image_command,
    tts as tts_command,
)
from .message_handlers import (
    transcribe as transcribe_handler,
    vision as vision_handler,
    prompt as prompt_handler,
    inline_query as inline_query_handler,
    send_inline_query_result as send_inline_query_result_handler,
    handle_callback_inline_query as handle_callback_inline_query_handler,
)
from .setup_helpers import (
    check_allowed_and_within_budget as check_allowed_and_within_budget_helper,
    send_disallowed_message as send_disallowed_message_helper,
    send_budget_reached_message as send_budget_reached_message_helper,
    post_init as post_init_helper,
    run as run_helper,
)


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        self.config = config
        self.openai = openai
        bot_language = self.config['bot_language']
        self.commands = [
            BotCommand(command='help', description=localized_text('help_description', bot_language)),
            BotCommand(command='reset', description=localized_text('reset_description', bot_language)),
            BotCommand(command='stats', description=localized_text('stats_description', bot_language)),
            BotCommand(command='resend', description=localized_text('resend_description', bot_language))
        ]
        # If imaging is enabled, add the "image" command to the list
        if self.config.get('enable_image_generation', False):
            self.commands.append(BotCommand(command='image', description=localized_text('image_description', bot_language)))

        if self.config.get('enable_tts_generation', False):
            self.commands.append(BotCommand(command='tts', description=localized_text('tts_description', bot_language)))

        self.group_commands = [BotCommand(
            command='chat', description=localized_text('chat_description', bot_language)
        )] + self.commands
        self.disallowed_message = localized_text('disallowed', bot_language)
        self.budget_limit_message = localized_text('budget_limit', bot_language)
        self.usage = {}
        self.last_message = {}
        self.inline_queries_cache = {}

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await help_command(self, update, context)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await stats_command(self, update, context)

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await resend_command(self, update, context)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await reset_command(self, update, context)

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await image_command(self, update, context)

    async def tts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await tts_command(self, update, context)

    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await transcribe_handler(self, update, context)

    async def vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await vision_handler(self, update, context)

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await prompt_handler(self, update, context)

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await inline_query_handler(self, update, context)

    async def send_inline_query_result(self, update: Update, result_id, message_content, callback_data=""):
        await send_inline_query_result_handler(self, update, result_id, message_content, callback_data)

    async def handle_callback_inline_query(self, update: Update, context: CallbackContext):
        await handle_callback_inline_query_handler(self, update, context)

    async def check_allowed_and_within_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_inline=False) -> bool:
        return await check_allowed_and_within_budget_helper(self, update, context, is_inline)

    async def send_disallowed_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_inline=False):
        await send_disallowed_message_helper(self, update, context, is_inline)

    async def send_budget_reached_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_inline=False):
        await send_budget_reached_message_helper(self, update, context, is_inline)

    async def post_init(self, application: Application) -> None:
        await post_init_helper(self, application)

    def run(self):
        run_helper(self)

