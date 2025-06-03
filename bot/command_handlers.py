"""Command handler functions for the ChatGPT Telegram bot."""

from __future__ import annotations

from uuid import uuid4
import asyncio
import logging
import os
import io

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle
from telegram import InputTextMessageContent, BotCommand, constants
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import ContextTypes

from utils import (
    is_group_chat,
    get_thread_id,
    message_text,
    wrap_with_indicator,
    split_into_chunks,
    edit_message_with_retry,
    get_stream_cutoff_values,
    is_allowed,
    get_remaining_budget,
    is_admin,
    is_within_budget,
    get_reply_to_message_id,
    add_chat_request_to_usage_tracker,
    error_handler,
    is_direct_result,
    handle_direct_result,
    cleanup_intermediate_files,
)
from openai_helper import localized_text
from usage_tracker import UsageTracker


async def help(bot, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the help menu."""
    commands = bot.group_commands if is_group_chat(update) else bot.commands
    commands_description = [f'/{command.command} - {command.description}' for command in commands]
    bot_language = bot.config['bot_language']
    help_text = (
        localized_text('help_text', bot_language)[0]
        + '\n\n'
        + '\n'.join(commands_description)
        + '\n\n'
        + localized_text('help_text', bot_language)[1]
        + '\n\n'
        + localized_text('help_text', bot_language)[2]
    )
    await update.message.reply_text(help_text, disable_web_page_preview=True)

async def stats(bot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Returns token usage statistics for current day and month.
    """
    if not await is_allowed(bot.config, update, context):
        logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                        'is not allowed to request their usage statistics')
        await bot.send_disallowed_message(update, context)
        return

    logging.info(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                 'requested their usage statistics')

    user_id = update.message.from_user.id
    if user_id not in bot.usage:
        bot.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

    tokens_today, tokens_month = bot.usage[user_id].get_current_token_usage()
    images_today, images_month = bot.usage[user_id].get_current_image_count()
    (transcribe_minutes_today, transcribe_seconds_today, transcribe_minutes_month,
     transcribe_seconds_month) = bot.usage[user_id].get_current_transcription_duration()
    vision_today, vision_month = bot.usage[user_id].get_current_vision_tokens()
    characters_today, characters_month = bot.usage[user_id].get_current_tts_usage()
    current_cost = bot.usage[user_id].get_current_cost()

    chat_id = update.effective_chat.id
    chat_messages, chat_token_length = bot.openai.get_conversation_stats(chat_id)
    remaining_budget = get_remaining_budget(bot.config, bot.usage, update)
    bot_language = bot.config['bot_language']
    
    text_current_conversation = (
        f"*{localized_text('stats_conversation', bot_language)[0]}*:\n"
        f"{chat_messages} {localized_text('stats_conversation', bot_language)[1]}\n"
        f"{chat_token_length} {localized_text('stats_conversation', bot_language)[2]}\n"
        "----------------------------\n"
    )
    
    # Check if image generation is enabled and, if so, generate the image statistics for today
    text_today_images = ""
    if bot.config.get('enable_image_generation', False):
        text_today_images = f"{images_today} {localized_text('stats_images', bot_language)}\n"

    text_today_vision = ""
    if bot.config.get('enable_vision', False):
        text_today_vision = f"{vision_today} {localized_text('stats_vision', bot_language)}\n"

    text_today_tts = ""
    if bot.config.get('enable_tts_generation', False):
        text_today_tts = f"{characters_today} {localized_text('stats_tts', bot_language)}\n"
    
    text_today = (
        f"*{localized_text('usage_today', bot_language)}:*\n"
        f"{tokens_today} {localized_text('stats_tokens', bot_language)}\n"
        f"{text_today_images}"  # Include the image statistics for today if applicable
        f"{text_today_vision}"
        f"{text_today_tts}"
        f"{transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
        f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
        f"{localized_text('stats_total', bot_language)}{current_cost['cost_today']:.2f}\n"
        "----------------------------\n"
    )
    
    text_month_images = ""
    if bot.config.get('enable_image_generation', False):
        text_month_images = f"{images_month} {localized_text('stats_images', bot_language)}\n"

    text_month_vision = ""
    if bot.config.get('enable_vision', False):
        text_month_vision = f"{vision_month} {localized_text('stats_vision', bot_language)}\n"

    text_month_tts = ""
    if bot.config.get('enable_tts_generation', False):
        text_month_tts = f"{characters_month} {localized_text('stats_tts', bot_language)}\n"
    
    # Check if image generation is enabled and, if so, generate the image statistics for the month
    text_month = (
        f"*{localized_text('usage_month', bot_language)}:*\n"
        f"{tokens_month} {localized_text('stats_tokens', bot_language)}\n"
        f"{text_month_images}"  # Include the image statistics for the month if applicable
        f"{text_month_vision}"
        f"{text_month_tts}"
        f"{transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
        f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
        f"{localized_text('stats_total', bot_language)}{current_cost['cost_month']:.2f}"
    )

    # text_budget filled with conditional content
    text_budget = "\n\n"
    budget_period = bot.config['budget_period']
    if remaining_budget < float('inf'):
        text_budget += (
            f"{localized_text('stats_budget', bot_language)}"
            f"{localized_text(budget_period, bot_language)}: "
            f"${remaining_budget:.2f}.\n"
        )
    # No longer works as of July 21st 2023, as OpenAI has removed the billing API
    # add OpenAI account information for admin request
    # if is_admin(bot.config, user_id):
    #     text_budget += (
    #         f"{localized_text('stats_openai', bot_language)}"
    #         f"{bot.openai.get_billing_current_month():.2f}"
    #     )

    usage_text = text_current_conversation + text_today + text_month + text_budget
    await update.message.reply_text(usage_text, parse_mode=constants.ParseMode.MARKDOWN)

async def resend(bot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Resend the last request
    """
    if not await is_allowed(bot.config, update, context):
        logging.warning(f'User {update.message.from_user.name}  (id: {update.message.from_user.id})'
                        ' is not allowed to resend the message')
        await bot.send_disallowed_message(update, context)
        return

    chat_id = update.effective_chat.id
    if chat_id not in bot.last_message:
        logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id})'
                        ' does not have anything to resend')
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=localized_text('resend_failed', bot.config['bot_language'])
        )
        return

    # Update message text, clear bot.last_message and send the request to prompt
    logging.info(f'Resending the last prompt from user: {update.message.from_user.name} '
                 f'(id: {update.message.from_user.id})')
    with update.message._unfrozen() as message:
        message.text = bot.last_message.pop(chat_id)

    await bot.prompt(update=update, context=context)

async def reset(bot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Resets the conversation.
    """
    if not await is_allowed(bot.config, update, context):
        logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                        'is not allowed to reset the conversation')
        await bot.send_disallowed_message(update, context)
        return

    logging.info(f'Resetting the conversation for user {update.message.from_user.name} '
                 f'(id: {update.message.from_user.id})...')

    chat_id = update.effective_chat.id
    reset_content = message_text(update.message)
    bot.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
    await update.effective_message.reply_text(
        message_thread_id=get_thread_id(update),
        text=localized_text('reset_done', bot.config['bot_language'])
    )

async def image(bot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Generates an image for the given prompt using DALL·E APIs
    """
    if not bot.config['enable_image_generation'] \
            or not await bot.check_allowed_and_within_budget(update, context):
        return

    image_query = message_text(update.message)
    if image_query == '':
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=localized_text('image_no_prompt', bot.config['bot_language'])
        )
        return

    logging.info(f'New image generation request received from user {update.message.from_user.name} '
                 f'(id: {update.message.from_user.id})')

    async def _generate():
        try:
            image_url, image_size = await bot.openai.generate_image(prompt=image_query)
            if bot.config['image_receive_mode'] == 'photo':
                await update.effective_message.reply_photo(
                    reply_to_message_id=get_reply_to_message_id(bot.config, update),
                    photo=image_url
                )
            elif bot.config['image_receive_mode'] == 'document':
                await update.effective_message.reply_document(
                    reply_to_message_id=get_reply_to_message_id(bot.config, update),
                    document=image_url
                )
            else:
                raise Exception(f"env variable IMAGE_RECEIVE_MODE has invalid value {bot.config['image_receive_mode']}")
            # add image request to users usage tracker
            user_id = update.message.from_user.id
            bot.usage[user_id].add_image_request(image_size, bot.config['image_prices'])
            # add guest chat request to guest usage tracker
            if str(user_id) not in bot.config['allowed_user_ids'].split(',') and 'guests' in bot.usage:
                bot.usage["guests"].add_image_request(image_size, bot.config['image_prices'])

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(bot.config, update),
                text=f"{localized_text('image_fail', bot.config['bot_language'])}: {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN
            )

    await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_PHOTO)

async def tts(bot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Generates an speech for the given input using TTS APIs
    """
    if not bot.config['enable_tts_generation'] \
            or not await bot.check_allowed_and_within_budget(update, context):
        return

    tts_query = message_text(update.message)
    if tts_query == '':
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=localized_text('tts_no_prompt', bot.config['bot_language'])
        )
        return

    logging.info(f'New speech generation request received from user {update.message.from_user.name} '
                 f'(id: {update.message.from_user.id})')

    async def _generate():
        try:
            speech_file, text_length = await bot.openai.generate_speech(text=tts_query)

            await update.effective_message.reply_voice(
                reply_to_message_id=get_reply_to_message_id(bot.config, update),
                voice=speech_file
            )
            speech_file.close()
            # add image request to users usage tracker
            user_id = update.message.from_user.id
            bot.usage[user_id].add_tts_request(text_length, bot.config['tts_model'], bot.config['tts_prices'])
            # add guest chat request to guest usage tracker
            if str(user_id) not in bot.config['allowed_user_ids'].split(',') and 'guests' in bot.usage:
                bot.usage["guests"].add_tts_request(text_length, bot.config['tts_model'], bot.config['tts_prices'])

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(bot.config, update),
                text=f"{localized_text('tts_fail', bot.config['bot_language'])}: {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN
            )

    await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_VOICE)

