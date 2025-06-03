"""Message handler functions for the ChatGPT Telegram bot."""

from __future__ import annotations

from uuid import uuid4
import asyncio
import logging
import os
import io

from telegram import Update, constants, InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle
from telegram import InputTextMessageContent
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import ContextTypes, CallbackContext
from pydub import AudioSegment
from PIL import Image

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

async def transcribe(bot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Transcribe audio messages.
    """
    if not bot.config['enable_transcription'] or not await bot.check_allowed_and_within_budget(update, context):
        return

    if is_group_chat(update) and bot.config['ignore_group_transcriptions']:
        logging.info('Transcription coming from group chat, ignoring...')
        return

    chat_id = update.effective_chat.id
    filename = update.message.effective_attachment.file_unique_id

    async def _execute():
        filename_mp3 = f'{filename}.mp3'
        bot_language = bot.config['bot_language']
        try:
            media_file = await context.bot.get_file(update.message.effective_attachment.file_id)
            await media_file.download_to_drive(filename)
        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(bot.config, update),
                text=(
                    f"{localized_text('media_download_fail', bot_language)[0]}: "
                    f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                ),
                parse_mode=constants.ParseMode.MARKDOWN
            )
            return

        try:
            audio_track = AudioSegment.from_file(filename)
            audio_track.export(filename_mp3, format="mp3")
            logging.info(f'New transcribe request received from user {update.message.from_user.name} '
                         f'(id: {update.message.from_user.id})')

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(bot.config, update),
                text=localized_text('media_type_fail', bot_language)
            )
            if os.path.exists(filename):
                os.remove(filename)
            return

        user_id = update.message.from_user.id
        if user_id not in bot.usage:
            bot.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        try:
            transcript = await bot.openai.transcribe(filename_mp3)

            transcription_price = bot.config['transcription_price']
            bot.usage[user_id].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

            allowed_user_ids = bot.config['allowed_user_ids'].split(',')
            if str(user_id) not in allowed_user_ids and 'guests' in bot.usage:
                bot.usage["guests"].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

            # check if transcript starts with any of the prefixes
            response_to_transcription = any(transcript.lower().startswith(prefix.lower()) if prefix else False
                                            for prefix in bot.config['voice_reply_prompts'])

            if bot.config['voice_reply_transcript'] and not response_to_transcription:

                # Split into chunks of 4096 characters (Telegram's message limit)
                transcript_output = f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\""
                chunks = split_into_chunks(transcript_output)

                for index, transcript_chunk in enumerate(chunks):
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(bot.config, update) if index == 0 else None,
                        text=transcript_chunk,
                        parse_mode=constants.ParseMode.MARKDOWN
                    )
            else:
                # Get the response of the transcript
                response, total_tokens = await bot.openai.get_chat_response(chat_id=chat_id, query=transcript)

                bot.usage[user_id].add_chat_tokens(total_tokens, bot.config['token_price'])
                if str(user_id) not in allowed_user_ids and 'guests' in bot.usage:
                    bot.usage["guests"].add_chat_tokens(total_tokens, bot.config['token_price'])

                # Split into chunks of 4096 characters (Telegram's message limit)
                transcript_output = (
                    f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\"\n\n"
                    f"_{localized_text('answer', bot_language)}:_\n{response}"
                )
                chunks = split_into_chunks(transcript_output)

                for index, transcript_chunk in enumerate(chunks):
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(bot.config, update) if index == 0 else None,
                        text=transcript_chunk,
                        parse_mode=constants.ParseMode.MARKDOWN
                    )

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(bot.config, update),
                text=f"{localized_text('transcribe_fail', bot_language)}: {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN
            )
        finally:
            if os.path.exists(filename_mp3):
                os.remove(filename_mp3)
            if os.path.exists(filename):
                os.remove(filename)

    await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)

async def vision(bot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Interpret image using vision model.
    """
    if not bot.config['enable_vision'] or not await bot.check_allowed_and_within_budget(update, context):
        return

    chat_id = update.effective_chat.id
    prompt = update.message.caption

    if is_group_chat(update):
        if bot.config['ignore_group_vision']:
            logging.info('Vision coming from group chat, ignoring...')
            return
        else:
            trigger_keyword = bot.config['group_trigger_keyword']
            if (prompt is None and trigger_keyword != '') or \
               (prompt is not None and not prompt.lower().startswith(trigger_keyword.lower())):
                logging.info('Vision coming from group chat with wrong keyword, ignoring...')
                return
    
    image = update.message.effective_attachment[-1]
    

    async def _execute():
        bot_language = bot.config['bot_language']
        try:
            media_file = await context.bot.get_file(image.file_id)
            temp_file = io.BytesIO(await media_file.download_as_bytearray())
        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(bot.config, update),
                text=(
                    f"{localized_text('media_download_fail', bot_language)[0]}: "
                    f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                ),
                parse_mode=constants.ParseMode.MARKDOWN
            )
            return
        
        # convert jpg from telegram to png as understood by openai

        temp_file_png = io.BytesIO()

        try:
            original_image = Image.open(temp_file)
            
            original_image.save(temp_file_png, format='PNG')
            logging.info(f'New vision request received from user {update.message.from_user.name} '
                         f'(id: {update.message.from_user.id})')

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(bot.config, update),
                text=localized_text('media_type_fail', bot_language)
            )
        
        

        user_id = update.message.from_user.id
        if user_id not in bot.usage:
            bot.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        if bot.config['stream']:

            stream_response = bot.openai.interpret_image_stream(chat_id=chat_id, fileobj=temp_file_png, prompt=prompt)
            i = 0
            prev = ''
            sent_message = None
            backoff = 0
            stream_chunk = 0

            async for content, tokens in stream_response:
                if is_direct_result(content):
                    return await handle_direct_result(bot.config, update, content)

                if len(content.strip()) == 0:
                    continue

                stream_chunks = split_into_chunks(content)
                if len(stream_chunks) > 1:
                    content = stream_chunks[-1]
                    if stream_chunk != len(stream_chunks) - 1:
                        stream_chunk += 1
                        try:
                            await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                          stream_chunks[-2])
                        except:
                            pass
                        try:
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                text=content if len(content) > 0 else "..."
                            )
                        except:
                            pass
                        continue

                cutoff = get_stream_cutoff_values(update, content)
                cutoff += backoff

                if i == 0:
                    try:
                        if sent_message is not None:
                            await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                             message_id=sent_message.message_id)
                        sent_message = await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(bot.config, update),
                            text=content,
                        )
                    except:
                        continue

                elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                    prev = content

                    try:
                        use_markdown = tokens != 'not_finished'
                        await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                      text=content, markdown=use_markdown)

                    except RetryAfter as e:
                        backoff += 5
                        await asyncio.sleep(e.retry_after)
                        continue

                    except TimedOut:
                        backoff += 5
                        await asyncio.sleep(0.5)
                        continue

                    except Exception:
                        backoff += 5
                        continue

                    await asyncio.sleep(0.01)

                i += 1
                if tokens != 'not_finished':
                    total_tokens = int(tokens)

            
        else:

            try:
                interpretation, total_tokens = await bot.openai.interpret_image(chat_id, temp_file_png, prompt=prompt)


                try:
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(bot.config, update),
                        text=interpretation,
                        parse_mode=constants.ParseMode.MARKDOWN
                    )
                except BadRequest:
                    try:
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(bot.config, update),
                            text=interpretation
                        )
                    except Exception as e:
                        logging.exception(e)
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(bot.config, update),
                            text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                            parse_mode=constants.ParseMode.MARKDOWN
                        )
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(bot.config, update),
                    text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )
        vision_token_price = bot.config['vision_token_price']
        bot.usage[user_id].add_vision_tokens(total_tokens, vision_token_price)

        allowed_user_ids = bot.config['allowed_user_ids'].split(',')
        if str(user_id) not in allowed_user_ids and 'guests' in bot.usage:
            bot.usage["guests"].add_vision_tokens(total_tokens, vision_token_price)

    await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)

async def prompt(bot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    React to incoming messages and respond accordingly.
    """
    if update.edited_message or not update.message or update.message.via_bot:
        return

    if not await bot.check_allowed_and_within_budget(update, context):
        return

    logging.info(
        f'New message received from user {update.message.from_user.name} (id: {update.message.from_user.id})')
    chat_id = update.effective_chat.id
    user_id = update.message.from_user.id
    prompt = message_text(update.message)
    bot.last_message[chat_id] = prompt

    if is_group_chat(update):
        trigger_keyword = bot.config['group_trigger_keyword']

        if prompt.lower().startswith(trigger_keyword.lower()) or update.message.text.lower().startswith('/chat'):
            if prompt.lower().startswith(trigger_keyword.lower()):
                prompt = prompt[len(trigger_keyword):].strip()

            if update.message.reply_to_message and \
                    update.message.reply_to_message.text and \
                    update.message.reply_to_message.from_user.id != context.bot.id:
                prompt = f'"{update.message.reply_to_message.text}" {prompt}'
        else:
            if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                logging.info('Message is a reply to the bot, allowing...')
            else:
                logging.warning('Message does not start with trigger keyword, ignoring...')
                return

    try:
        total_tokens = 0

        if bot.config['stream']:
            await update.effective_message.reply_chat_action(
                action=constants.ChatAction.TYPING,
                message_thread_id=get_thread_id(update)
            )

            stream_response = bot.openai.get_chat_response_stream(chat_id=chat_id, query=prompt)
            i = 0
            prev = ''
            sent_message = None
            backoff = 0
            stream_chunk = 0

            async for content, tokens in stream_response:
                if is_direct_result(content):
                    return await handle_direct_result(bot.config, update, content)

                if len(content.strip()) == 0:
                    continue

                stream_chunks = split_into_chunks(content)
                if len(stream_chunks) > 1:
                    content = stream_chunks[-1]
                    if stream_chunk != len(stream_chunks) - 1:
                        stream_chunk += 1
                        try:
                            await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                          stream_chunks[-2])
                        except:
                            pass
                        try:
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                text=content if len(content) > 0 else "..."
                            )
                        except:
                            pass
                        continue

                cutoff = get_stream_cutoff_values(update, content)
                cutoff += backoff

                if i == 0:
                    try:
                        if sent_message is not None:
                            await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                             message_id=sent_message.message_id)
                        sent_message = await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(bot.config, update),
                            text=content,
                        )
                    except:
                        continue

                elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                    prev = content

                    try:
                        use_markdown = tokens != 'not_finished'
                        await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                      text=content, markdown=use_markdown)

                    except RetryAfter as e:
                        backoff += 5
                        await asyncio.sleep(e.retry_after)
                        continue

                    except TimedOut:
                        backoff += 5
                        await asyncio.sleep(0.5)
                        continue

                    except Exception:
                        backoff += 5
                        continue

                    await asyncio.sleep(0.01)

                i += 1
                if tokens != 'not_finished':
                    total_tokens = int(tokens)

        else:
            async def _reply():
                nonlocal total_tokens
                response, total_tokens = await bot.openai.get_chat_response(chat_id=chat_id, query=prompt)

                if is_direct_result(response):
                    return await handle_direct_result(bot.config, update, response)

                # Split into chunks of 4096 characters (Telegram's message limit)
                chunks = split_into_chunks(response)

                for index, chunk in enumerate(chunks):
                    try:
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(bot.config,
                                                                        update) if index == 0 else None,
                            text=chunk,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )
                    except Exception:
                        try:
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(bot.config,
                                                                            update) if index == 0 else None,
                                text=chunk
                            )
                        except Exception as exception:
                            raise exception

            await wrap_with_indicator(update, context, _reply, constants.ChatAction.TYPING)

        add_chat_request_to_usage_tracker(bot.usage, bot.config, user_id, total_tokens)

    except Exception as e:
        logging.exception(e)
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            reply_to_message_id=get_reply_to_message_id(bot.config, update),
            text=f"{localized_text('chat_fail', bot.config['bot_language'])} {str(e)}",
            parse_mode=constants.ParseMode.MARKDOWN
        )

async def inline_query(bot, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the inline query. This is run when you type: @botusername <query>
    """
    query = update.inline_query.query
    if len(query) < 3:
        return
    if not await bot.check_allowed_and_within_budget(update, context, is_inline=True):
        return

    callback_data_suffix = "gpt:"
    result_id = str(uuid4())
    bot.inline_queries_cache[result_id] = query
    callback_data = f'{callback_data_suffix}{result_id}'

    await bot.send_inline_query_result(update, result_id, message_content=query, callback_data=callback_data)

async def send_inline_query_result(bot, update: Update, result_id, message_content, callback_data=""):
    """
    Send inline query result
    """
    try:
        reply_markup = None
        bot_language = bot.config['bot_language']
        if callback_data:
            reply_markup = InlineKeyboardMarkup([[
                InlineKeyboardButton(text=f'🤖 {localized_text("answer_with_chatgpt", bot_language)}',
                                     callback_data=callback_data)
            ]])

        inline_query_result = InlineQueryResultArticle(
            id=result_id,
            title=localized_text("ask_chatgpt", bot_language),
            input_message_content=InputTextMessageContent(message_content),
            description=message_content,
            thumbnail_url='https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea-b02a7a32149a.png',
            reply_markup=reply_markup
        )

        await update.inline_query.answer([inline_query_result], cache_time=0)
    except Exception as e:
        logging.error(f'An error occurred while generating the result card for inline query {e}')

async def handle_callback_inline_query(bot, update: Update, context: CallbackContext):
    """
    Handle the callback query from the inline query result
    """
    callback_data = update.callback_query.data
    user_id = update.callback_query.from_user.id
    inline_message_id = update.callback_query.inline_message_id
    name = update.callback_query.from_user.name
    callback_data_suffix = "gpt:"
    query = ""
    bot_language = bot.config['bot_language']
    answer_tr = localized_text("answer", bot_language)
    loading_tr = localized_text("loading", bot_language)

    try:
        if callback_data.startswith(callback_data_suffix):
            unique_id = callback_data.split(':')[1]
            total_tokens = 0

            # Retrieve the prompt from the cache
            query = bot.inline_queries_cache.get(unique_id)
            if query:
                bot.inline_queries_cache.pop(unique_id)
            else:
                error_message = (
                    f'{localized_text("error", bot_language)}. '
                    f'{localized_text("try_again", bot_language)}'
                )
                await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                              text=f'{query}\n\n_{answer_tr}:_\n{error_message}',
                                              is_inline=True)
                return

            unavailable_message = localized_text("function_unavailable_in_inline_mode", bot_language)
            if bot.config['stream']:
                stream_response = bot.openai.get_chat_response_stream(chat_id=user_id, query=query)
                i = 0
                prev = ''
                backoff = 0
                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        cleanup_intermediate_files(content)
                        await edit_message_with_retry(context, chat_id=None,
                                                      message_id=inline_message_id,
                                                      text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
                                                      is_inline=True)
                        return

                    if len(content.strip()) == 0:
                        continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            await edit_message_with_retry(context, chat_id=None,
                                                          message_id=inline_message_id,
                                                          text=f'{query}\n\n{answer_tr}:\n{content}',
                                                          is_inline=True)
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content
                        try:
                            use_markdown = tokens != 'not_finished'
                            divider = '_' if use_markdown else ''
                            text = f'{query}\n\n{divider}{answer_tr}:{divider}\n{content}'

                            # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                            text = text[:4096]

                            await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                          text=text, markdown=use_markdown, is_inline=True)

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue
                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue
                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:
                async def _send_inline_query_response():
                    nonlocal total_tokens
                    # Edit the current message to indicate that the answer is being processed
                    await context.bot.edit_message_text(inline_message_id=inline_message_id,
                                                        text=f'{query}\n\n_{answer_tr}:_\n{loading_tr}',
                                                        parse_mode=constants.ParseMode.MARKDOWN)

                    logging.info(f'Generating response for inline query by {name}')
                    response, total_tokens = await bot.openai.get_chat_response(chat_id=user_id, query=query)

                    if is_direct_result(response):
                        cleanup_intermediate_files(response)
                        await edit_message_with_retry(context, chat_id=None,
                                                      message_id=inline_message_id,
                                                      text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
                                                      is_inline=True)
                        return

                    text_content = f'{query}\n\n_{answer_tr}:_\n{response}'

                    # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                    text_content = text_content[:4096]

                    # Edit the original message with the generated content
                    await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                  text=text_content, is_inline=True)

                await wrap_with_indicator(update, context, _send_inline_query_response,
                                          constants.ChatAction.TYPING, is_inline=True)

            add_chat_request_to_usage_tracker(bot.usage, bot.config, user_id, total_tokens)

    except Exception as e:
        logging.error(f'Failed to respond to an inline query via button callback: {e}')
        logging.exception(e)
        localized_answer = localized_text('chat_fail', bot.config['bot_language'])
        await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                      text=f"{query}\n\n_{answer_tr}:_\n{localized_answer} {str(e)}",
                                      is_inline=True)

